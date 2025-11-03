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
    args = parser.parse_args()
    return args


def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image '{args.image_path}' not found. Place a PNG at this path or change image_path.")
    img = Image.open(args.image_path)
    coords, pixels, (H, W) = image_to_coords_and_pixels(img)
    N = coords.shape[0]
    logging.info(f"Coords/pixels shapes: {coords.shape}, {pixels.shape} (H={H}, W={W}, N={N})")

    dataset = TensorDataset(coords, pixels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    cfg = load_config(args.model_config)
    model = build_model_from_cfg(cfg)
    model = model.to(device)

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
            model.load_state_dict(resumed_ckpt["model_state_dict"])
        else:
            logging.warning("Checkpoint does not contain model_state_dict.")

        if "optimizer_state_dict" in resumed_ckpt:
            try:
                optimizer.load_state_dict(resumed_ckpt["optimizer_state_dict"])
                move_optimizer_state_to_device(optimizer, device)
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

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        for bc, bt in loader:
            optimizer.zero_grad()

            bc = bc.to(device, non_blocking=True)
            bt = bt.to(device, non_blocking=True)

            pred = model(bc)
            loss = criterion(pred, bt)

            loss.backward()
            optimizer.step()

            running += loss.item() * bc.shape[0]

        epoch_loss = running / N
        logging.info(f"Epoch {epoch:4d} loss {epoch_loss:.6f}")

        if use_wandb and wandb_run:
            try:
                wandb.log({"train/loss": epoch_loss}, step=epoch)
            except Exception as e:
                logging.warning(f"Failed to log to wandb: {e}")

        if epoch % args.save_ckpt_every == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "run_name": args.run_name,
                "wandb_run_id": (wandb_run.id if wandb_run else None),
            }
            ckpt_name = f"{args.run_name}_epoch_{epoch}.pt"
            torch.save(checkpoint, ckpt_name)
            logging.info(f"Saved checkpoint: {ckpt_name}")

        if epoch % args.save_img_every == 0:
            with torch.no_grad():
                out = infer_in_chunks(model, coords, device, chunk_size=args.inference_chunk_size)
                recon = (out.reshape(H, W, 3) * 255.0).astype(np.uint8)
                pil_recon = Image.fromarray(recon)
                img_name = f"{args.run_name}_epoch_{epoch}.png"
                pil_recon.save(img_name)
                logging.info(f"Saved reconstruction: {img_name}")
                if use_wandb and wandb_run:
                    try:
                        import wandb
                        wandb_run.log({"reconstruction": wandb.Image(pil_recon, caption=f"epoch {epoch}")}, step=epoch)
                    except Exception as e:
                        logging.warning(f"Failed to log reconstruction to wandb: {e}")

    # Final reconstruction (inference)
    with torch.no_grad():
        out = infer_in_chunks(model, coords, device, chunk_size=args.inference_chunk_size)
        recon = (out.reshape(H, W, 3) * 255.0).astype(np.uint8)
        pil_recon = Image.fromarray(recon)
        final_name = f"final_{args.run_name}_512.png"
        pil_recon.save(final_name)
        logging.info(f"Saved {final_name}")
        if use_wandb and wandb_run:
            try:
                import wandb
                wandb_run.log({"reconstruction/final": wandb.Image(pil_recon, caption="final")})
                wandb_run.finish()
            except Exception as e:
                logging.warning(f"Failed to finalize wandb run: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
