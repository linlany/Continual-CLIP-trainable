import os
import json
import hydra
import logging
from omegaconf import DictConfig

import torch
import statistics
from torch.utils.data import DataLoader, DistributedSampler
from continuum.metrics import Logger
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from continual_clip import utils
from continual_clip.models import load_model
from continual_clip.cls_datasets import build_cl_scenarios

import clip
import sys


def run_class_incremental(cfg, device, rank, world_size):

    cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, cfg.class_order))
    model = load_model(cfg, device)
    model = DDP(model, device_ids=[device.index], output_device=device.index)  # 使用 local_rank 绑定设备
    eval_dataset, classes_names = build_cl_scenarios(
        cfg, is_train=False, transforms=model.module.transforms
    )
    model.module.classes_names = classes_names
    train_dataset, _ = build_cl_scenarios(
        cfg, is_train=True, transforms=model.module.transforms)

    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = CrossEntropyLoss()
    epochs = cfg.epochs
    
    acc_list = []
    metric_logger = Logger(list_subsets=["test"])
    for task_id, _ in enumerate(eval_dataset):
        train_sampler = DistributedSampler(train_dataset[task_id], num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_dataset[task_id], batch_size=cfg.train_batch_size, sampler=train_sampler, num_workers=cfg.num_workers)

        if rank == 0:
            logging.info(f"Training for task {task_id} has started.")
        model.module.adaptation(task_id)
        for epoch in range(epochs):
            model.train()
            train_sampler.set_epoch(epoch)  # 设置分布式采样器的 epoch
            if rank == 0:
                logging.info(f"Task {task_id} Epoch {epoch+1}/{epochs} training started.")
            for batch_idx, (inputs, targets, t) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs, test=False)["logits_per_image"]
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                if rank == 0 and batch_idx % 10 == 0:
                    logging.info(f"Task {task_id} Epoch {epoch+1}/{epochs} Batch {batch_idx} Loss: {loss.item()}")

        if rank == 0:  # 仅主进程执行测试
            logging.info(f"Evaluation for task {task_id} has started.")

            model.eval()
            eval_loader = DataLoader(eval_dataset[:task_id + 1], batch_size=cfg.batch_size)
            for inputs, targets, task_ids in eval_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, test=True)['probs']
                metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")

            acc_list.append(100 * metric_logger.accuracy)
            with open(cfg.log_path, 'a+') as f:
                f.write(json.dumps({
                    'task': task_id,
                    'acc': round(100 * metric_logger.accuracy, 2),
                    'avg_acc': round(100 * metric_logger.average_incremental_accuracy, 2),
                    'forgetting': round(100 * metric_logger.forgetting, 6),
                    'acc_per_task': [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task],
                    'bwt': round(100 * metric_logger.backward_transfer, 2),
                    'fwt': round(100 * metric_logger.forward_transfer, 2),
                }) + '\n')
                metric_logger.end_task()

    if rank == 0:
        with open(cfg.log_path, 'a+') as f:
            f.write(json.dumps({
                'last': round(acc_list[-1], 2), 
                'avg': round(statistics.mean(acc_list), 2)
            }) + '\n')


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 获取本地 rank
    torch.cuda.set_device(local_rank)  # 设置本地 GPU
    return rank, world_size, local_rank


@hydra.main(config_path=None, config_name=None, version_base="1.1") 
def continual_clip(cfg: DictConfig) -> None:
    rank, world_size, local_rank = setup_distributed()
    cfg.workdir = utils.get_workdir(path=os.getcwd())
    cfg.dataset_root = os.path.join(cfg.workdir, cfg.dataset_root)

    if rank == 0:
        utils.save_config(cfg)
        with open(cfg.log_path, 'w+') as f: 
            pass
    device = torch.device("cuda", local_rank)  # 使用 local_rank 设置设备

    if cfg.scenario == "class":
        run_class_incremental(cfg, device, rank, world_size)

    else:
        if rank == 0:
            ValueError(f"You have entered `{cfg.scenario}` which is not a defined scenario, " 
                        "please choose from {{'class'}}.")

    dist.destroy_process_group()


if __name__ == "__main__":
    continual_clip()