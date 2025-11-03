import os
import random
import argparse
import logging
import yaml
import torch
import numpy as np

from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm

from registry import MODEL_REGISTRY
from utils import infer_in_chunks, image_to_coords_and_pixels, move_optimizer_state_to_device


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_model_from_cfg(cfg):
    name = cfg["model"]["name"]
    args = cfg["model"].get("args", {})
    ModelClass = MODEL_REGISTRY.get(name)
    if not ModelClass:
        raise KeyError(f"Model {name} not in registry")
    return ModelClass(**args)


def load_wandb_env():
    """
    Loads only the minimal WandB env vars from .env:
        - WANDB_API_KEY
        - WANDB_PROJECT
        - WANDB_ENTITY
    """
    load_dotenv()
    return {
        "api_key": os.getenv("WANDB_API_KEY"),
        "project": os.getenv("WANDB_PROJECT"),
        "entity": os.getenv("WANDB_ENTITY"),
        "run_name": os.getenv("WANDB_RUN_NAME"),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, help="Path to model config", required=True)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16_384, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20_000, help="Training epochs")
    parser.add_argument("--save_ckpt_every", type=int, default=200, help="Save model every X epochs")
    parser.add_argument("--save_img_every", type=int, default=10, help="Save image every X epochs")
    parser.add_argument("--inference_chunk_size", type=int, default=65_536, help="Chunk size of coords for image saving")
    parser.add_argument("--image_path", type=str, help="Path to the training image", required=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--run_name", type=str, default="hello_world", help="Run name for WandB and checkpointing")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--pin_memory", action="store_true", help="Use DataLoader pin_memory")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (torch.cuda.amp)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() if available")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--data_on_gpu", action="store_true", help="Load full dataset tensors onto GPU (fast but uses GPU memory)")
    parser.add_argument("--log_every", type=int, default=50, help="Log every N batches (0 to disable)")
    parser.add_argument("--ckpt_dir", type=str, default=".", help="Directory where checkpoints are saved")
    args = parser.parse_args()
    return args


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main(args):
    set_seed(args.seed)
    _ensure_dir(args.ckpt_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image '{args.image_path}' not found. Place a PNG/JPEG at this path or change image_path.")
    img = Image.open(args.image_path).convert("RGB")
    coords, pixels, (H, W) = image_to_coords_and_pixels(img)
    N = coords.shape[0]
    logging.info(f"Coords/pixels shapes: {coords.shape}, {pixels.shape} (H={H}, W={W}, N={N})")

    if args.data_on_gpu and device.type == "cuda":
        logging.info("Moving full dataset to GPU (data_on_gpu=True). This will use GPU memory.")
        coords = coords.to(device)
        pixels = pixels.to(device)
        dataset = TensorDataset(coords, pixels)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    else:
        dataset = TensorDataset(coords, pixels)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=args.pin_memory)

    cfg = load_config(args.model_config)
    model = build_model_from_cfg(cfg)
    model = model.to(device)

    if args.compile:
        try:
            model = torch.compile(model)  # may speed up
            logging.info("Applied torch.compile() to the model.")
        except Exception as e:
            logging.warning(f"torch.compile() failed or not available: {e}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    resumed_ckpt = None
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint '{args.resume}' not found.")
        logging.info(f"Loading checkpoint from {args.resume}")
        resumed_ckpt = torch.load(args.resume, map_location=device)

        if "model_state_dict" in resumed_ckpt:
            try:
                model.load_state_dict(resumed_ckpt["model_state_dict"])
                logging.info("Loaded model_state_dict from checkpoint.")
            except Exception as e:
                logging.warning(f"Failed to load model_state_dict cleanly: {e}")
        else:
            logging.warning("Checkpoint does not contain model_state_dict.")

        if "optimizer_state_dict" in resumed_ckpt:
            try:
                optimizer.load_state_dict(resumed_ckpt["optimizer_state_dict"])
                try:
                    move_optimizer_state_to_device(optimizer, device)
                except Exception:
                    move_optimizer_state_to_device(optimizer, device)
                logging.info("Loaded optimizer_state_dict and moved to device.")
            except Exception as e:
                logging.warning(f"Failed to fully load optimizer state: {e}")

        if "epoch" in resumed_ckpt:
            start_epoch = int(resumed_ckpt["epoch"]) + 1
            logging.info(f"Resuming from checkpoint epoch {resumed_ckpt['epoch']}, starting at epoch {start_epoch}")

        if "run_name" in resumed_ckpt and resumed_ckpt["run_name"]:
            logging.info(f"Resuming run_name from checkpoint: {resumed_ckpt['run_name']}")
            args.run_name = resumed_ckpt["run_name"]

    use_wandb = args.wandb
    wandb_run = None
    if use_wandb:
        try:
            import wandb
        except Exception as e:
            raise RuntimeError("wandb is required but not installed. Please install with `pip install wandb`.") from e

        wb = load_wandb_env()

        if wb["api_key"]:
            try:
                wandb.login(key=wb["api_key"])
            except Exception:
                logging.warning("Failed to login to wandb with WANDB_API_KEY. Continuing and attempting init (may fail).")

        if not wb["project"]:
            raise RuntimeError("WANDB_PROJECT environment variable must be set in the .env when using --wandb.")

        run_name = args.run_name or wb.get("run_name")
        if not run_name:
            model_name = cfg.get("model", {}).get("name", "run")
            run_name = f"{model_name}_bs_{args.batch_size}_lr_{args.lr}_seed_{args.seed}"

        init_kwargs = {"project": wb["project"], "name": run_name}
        if wb.get("entity"):
            init_kwargs["entity"] = wb["entity"]

        resume_init = None
        if resumed_ckpt and resumed_ckpt.get("wandb_run_id"):
            ckpt_wandb_id = resumed_ckpt.get("wandb_run_id")
            logging.info(f"Found wandb run id in checkpoint: {ckpt_wandb_id}. Attempting to resume WandB run.")
            init_kwargs["id"] = ckpt_wandb_id
            resume_init = "must"

        try:
            if resume_init:
                wandb_run = wandb.init(resume=resume_init, **init_kwargs)
            else:
                wandb_run = wandb.init(**init_kwargs)

            wandb.config.update({
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "image_path": args.image_path,
                "seed": args.seed,
                "save_ckpt_every": args.save_ckpt_every,
                "save_img_every": args.save_img_every,
                "inference_chunk_size": args.inference_chunk_size,
                "model_name": cfg.get("model", {}).get("name"),
                "model_args": cfg.get("model", {}).get("args", {}),
            }, allow_val_change=True)

            logging.info(f"Started wandb run (project={wb['project']})")
        except Exception as e:
            logging.warning(f"Failed to initialize wandb run: {e}")
            wandb_run = None
            use_wandb = False

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    total_steps_per_epoch = len(loader)
    logging.info(f"Starting training from epoch {start_epoch} to {args.epochs}. Steps per epoch: {total_steps_per_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss_sum = 0.0
        seen_samples = 0

        iterator = enumerate(loader, start=1)

        pbar = tqdm(iterator, total=total_steps_per_epoch, desc=f"Epoch {epoch}", leave=False)

        optimizer.zero_grad()
        for bidx, (bc, bt) in pbar:

            if not args.data_on_gpu:
                bc = bc.to(device, non_blocking=True)
                bt = bt.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                pred = model(bc)
                loss = criterion(pred, bt)

            loss_unscaled = loss.detach()
            loss_to_backward = loss / args.accum_steps

            # backward
            if args.amp and device.type == "cuda":
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            if (bidx % args.accum_steps) == 0:
                if args.amp and device.type == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            batch_n = bc.shape[0]
            running_loss_sum += float(loss_unscaled.item()) * batch_n
            seen_samples += batch_n

            avg_loss = running_loss_sum / seen_samples if seen_samples > 0 else float("nan")
            pbar.set_postfix_str(f"loss={avg_loss:.6f}")

            if args.log_every and (bidx % args.log_every == 0):
                logging.info(f"Epoch {epoch} batch {bidx}/{total_steps_per_epoch} avg_loss {avg_loss:.6f}")
                if use_wandb and wandb_run:
                    try:
                        wandb_run.log({"train/loss_batch": avg_loss}, step=(epoch - 1) * total_steps_per_epoch + bidx)
                    except Exception as e:
                        logging.warning(f"Failed to log batch loss to wandb: {e}")

        epoch_loss = running_loss_sum / N
        logging.info(f"Epoch {epoch:4d} loss {epoch_loss:.6f}")

        if use_wandb and wandb_run:
            try:
                wandb_run.log({"train/loss": epoch_loss}, step=epoch)
            except Exception as e:
                logging.warning(f"Failed to log epoch loss to wandb: {e}")

        if epoch % args.save_ckpt_every == 0 or epoch == args.epochs:
            try:
                model_state = model.state_dict()
            except Exception:
                try:
                    model_state = model._orig_mod.state_dict()
                except Exception:
                    model_state = None

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model_state if model_state is not None else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "run_name": args.run_name,
                "wandb_run_id": (wandb_run.id if wandb_run else None),
            }
            ckpt_name = os.path.join(args.ckpt_dir, f"{args.run_name}_epoch_{epoch}.pt")
            torch.save(checkpoint, ckpt_name)
            logging.info(f"Saved checkpoint: {ckpt_name}")

        if epoch % args.save_img_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                out = infer_in_chunks(model, coords, device, chunk_size=args.inference_chunk_size)
                recon = (out.reshape(H, W, 3) * 255.0).astype(np.uint8)
                pil_recon = Image.fromarray(recon)
                img_name = os.path.join(args.ckpt_dir, f"{args.run_name}_epoch_{epoch}.png")
                pil_recon.save(img_name)
                logging.info(f"Saved reconstruction: {img_name}")
                if use_wandb and wandb_run:
                    try:
                        import wandb as _wandb
                        wandb_run.log({"reconstruction": _wandb.Image(pil_recon, caption=f"epoch {epoch}")}, step=epoch)
                    except Exception as e:
                        logging.warning(f"Failed to log reconstruction to wandb: {e}")

    # Final reconstruction (inference)
    with torch.no_grad():
        out = infer_in_chunks(model, coords, device, chunk_size=args.inference_chunk_size)
        recon = (out.reshape(H, W, 3) * 255.0).astype(np.uint8)
        pil_recon = Image.fromarray(recon)
        final_name = os.path.join(args.ckpt_dir, f"final_{args.run_name}_512.png")
        pil_recon.save(final_name)
        logging.info(f"Saved {final_name}")
        if use_wandb and wandb_run:
            try:
                import wandb as _wandb
                wandb_run.log({"reconstruction/final": _wandb.Image(pil_recon, caption="final")})
                wandb_run.finish()
            except Exception as e:
                logging.warning(f"Failed to finalize wandb run: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)








