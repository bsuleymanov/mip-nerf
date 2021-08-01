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
import math
from torch import nn
import torch.multiprocessing as mp
from hydra import initialize, compose
from utils import namedtuple_map


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def learning_rate_decay(step, lr_init=5e-4, lr_final=5e-6, max_steps=1000000,
                        lr_delay_steps=0, lr_delay_mult=1):
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * torch.sin(
            0.5 * math.pi * torch.clamp(step / lr_delay_steps, 0, 1))
    else:
        delay_rate = 1.
    t = torch.clamp(step / max_steps, 0, 1)
    log_lerp = torch.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)
    return delay_rate * log_lerp



def run_training(rank, world_size, seed, cfg):
    set_seed(seed)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    print(f"{rank + 1}/{world_size} process initialized.")
    #if rank == 0:
    #    wandb.init(project=cfg.logging.project, entity=cfg.logging.entity)

    dataloader = instantiate(cfg.data.train.dataloader, world_size=world_size,
                             rank=rank).loader
    # test_dataloader = instantiate(cfg.data.validation.dataloader,
    #                               world_size=world_size, rank=rank).loader
    data_iter = iter(dataloader)
    step_per_epoch = len(dataloader)
    example_batch = next(data_iter)#.to(rank)
    example_rays = example_batch["rays"]
    example_rays = namedtuple_map(lambda x: x.to(rank), example_rays)
    example_rays = namedtuple_map(lambda x: torch.ones_like(x,
                                                            device=rank, dtype=torch.float32), example_rays)
    model, variables = make_mipnerf(
        example_rays,
        device=rank, randomized=cfg.training.randomized,
        white_bg=cfg.training.white_bg)
    model = DDP(model, device_ids=[rank], output_device=rank,
                find_unused_parameters=False)

    # weight decay in config
    # works with cuda only
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_parameters = sum([p.numel() for p in model_parameters])

    optimizer = instantiate(cfg.training.optimizer, params=[
        p for p in model.parameters() if p.requires_grad
    ], weight_decay=cfg.training.weight_decay_mult/model_parameters)

    start = 0
    for step in range(start, cfg.training.total_step):
        try:
            batch = next(data_iter)
        except:
            dataloader.sampler.set_epoch(step // step_per_epoch)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch_rays = namedtuple_map(lambda x: x.to(rank), batch["rays"])
        batch_pixels = batch["pixels"].to(rank)
        pred = model(batch_rays, cfg.training.randomized,
                     cfg.training.white_bg)
        mask = batch_rays.lossmult
        if cfg.training.disable_multiscale_loss:
            mask = torch.ones_like(mask)

        losses = []
        for (rgb, _, _) in pred:
            losses.append(
                (mask * (rgb - batch_pixels[..., :3]) ** 2).sum() / mask.sum())
        losses = torch.tensor(losses)
        losses = (cfg.training.coarse_loss_mult * losses[:-1].sum() + losses[-1])
        losses = losses.mean()

        if cfg.training.grad_max_val > 0:
            nn.utils.clip_grad_value_(model.parameters(), cfg.training.grad_max_val)

        if cfg.training.grad_max_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_max_norm)

        losses.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(losses)

def train_from_folder_distributed():
    #cfg = train_from_folder_distributed_subfunc()
    #print(cfg)
    world_size = torch.cuda.device_count()
    print(f"world_size: {world_size}")
    global_seed = 228
    initialize(config_path="configs", job_name="test_app")
    cfg = compose(config_name="full_experiment")
    mp.spawn(run_training,
             args=(world_size, global_seed, cfg),
             nprocs=torch.cuda.device_count(),
             join=True)



if __name__ == "__main__":
    train_from_folder_distributed()