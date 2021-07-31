import hydra
import torch
from hydra.utils import instantiate
import numpy as np
import random
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from models import make_mipnerf



def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def loss_fn():



def train_step(model, cfg, batch, optimizer):




def run_training(rank, world_size, seed, cfg):
    dataloader = instantiate(cfg.data.train.dataloader,
                             world_size=world_size, rank=rank)
    set_seed(seed)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    print(f"{rank + 1}/{world_size} process initialized.")
    if rank == 0:
        wandb.init(project=cfg.logging.project, entity=cfg.logging.entity)

    dataloader = instantiate(cfg.data.train.dataloader, world_size=world_size,
                             rank=rank).loader
    test_dataloader = instantiate(cfg.data.validation.dataloader,
                                  world_size=world_size, rank=rank).loader
    data_iter = iter(dataloader)
    example_batch = next(dataloader).to(rank)
    model, variables = make_mipnerf(
        torch.ones_like(example_batch, device=rank, dtype=torch.float32),
        device=rank)
    optimizer = instantiate(cfg.training.optimizer, params=[
        p for p in model.parameters() if p.requires_grad
    ])





@hydra.main("configs/", "full_experiment")
def main(cfg):
    ...




if __name__ == "__main__":
    main()