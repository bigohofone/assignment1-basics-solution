import os
import json
import glob
import random
import argparse
import numpy as np
import deepspeed
import wandb
import torch

from tqdm import tqdm

from cs336_basics.solutions.transformer_lm import TransformerLM
from cs336_basics.solutions.adamw import AdamW
from cs336_basics.solutions.cross_entropy import CrossEntropyLoss
from cs336_basics.solutions.learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics.solutions.gradient_clipping import clip_grad_norm_
from cs336_basics.solutions.data_loading import get_batch


def load_dataset(dataset_dir):
    pattern = os.path.join(dataset_dir, "*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No shard_w*.npy files found in {dataset_dir}")
    
    arrays = [np.load(f) for f in files]
    full_dataset = np.concatenate(arrays)
    return full_dataset

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser = deepspeed.add_config_arguments(parser)
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by deepspeed launcher")
parser.add_argument("--model_config_path", type=str, required=True)
parser.add_argument("--ds_config_path", type=str, required=True)
parser.add_argument("--dataset_dir", type=str, required=True)
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Dir to save checkpoints")
parser.add_argument("--save_interval", type=int, default=2000, help="Interval (in steps) to save checkpoints")
parser.add_argument("--resume_from", type=str, default=None, help="Path to a specific checkpoint to resume from")
# Training
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--min_lr", type=float, default=3e-5)
parser.add_argument("--context_length", type=int, default=256)
parser.add_argument("--total_iters", type=int, default=10000)
parser.add_argument("--warmup_iters", type=int, default=1000)
parser.add_argument("--z", type=float, default=0.0)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
# Wandb
parser.add_argument("--wandb_project", type=str, default="cs336-a1-ts")
parser.add_argument("--wandb_name", type=str, default="run-test")
args = parser.parse_args()


deepspeed.init_distributed()
rank = deepspeed.comm.get_rank()
seed_everything(42 + rank)


if rank == 0:
    wandb.login()
    wandb.init(project=args.wandb_project, name=args.wandb_name,
               config=vars(args), resume="allow")

full_dataset = load_dataset(args.dataset_dir)

with open(args.model_config_path, 'r') as f:
    model = TransformerLM(**json.load(f))

optimizer = AdamW(params=model.parameters(), lr=args.lr)

with open(args.ds_config_path, 'r') as f:
    ds_config = json.load(f)

model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    optimizer=optimizer,
    config=ds_config
)

loss_fn = CrossEntropyLoss(z=args.z)

start_step = 0
if args.resume_from:
    _, client_state = model_engine.load_checkpoint(args.resume_from)
    start_step = client_state.get('step', 0) + 1
    if rank == 0:
        print(f"Resuming training from step {start_step}")

pbar = tqdm(range(start_step, args.total_iters), disable=(rank != 0), desc="Training")

for step in pbar:
    current_lr = get_lr_cosine_schedule(
        it=step,
        max_learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.total_iters
    )

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    x_batch, y_batch = get_batch(
        dataset=full_dataset,
        batch_size=ds_config['train_micro_batch_size_per_gpu'],
        context_length=args.context_length,
        device=str(model_engine.device)
    )

    outputs = model_engine(x_batch)
    loss = loss_fn(outputs, y_batch)

    model_engine.backward(loss)

    clip_grad_norm_(
        parameters=model_engine.parameters(),
        max_l2_norm=args.max_grad_norm
    )

    model_engine.step()

    if rank == 0:
        wandb.log({
            "train/loss": loss.item(),
            "train/lr": current_lr,
            "train/step": step
        })
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

    if step > 0 and step % args.save_interval == 0:
        client_state = {'step': step}
        model_engine.save_checkpoint(args.checkpoint_dir, tag=f"step_{step}", client_state=client_state)
        if rank == 0:
            pbar.write(f"[Step {step}] Checkpoint saved.")

client_state = {'step': args.total_iters}
model_engine.save_checkpoint(args.checkpoint_dir, tag="final", client_state=client_state)

if rank == 0:
    print("Training finished.")
    wandb.finish()