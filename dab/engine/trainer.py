import sys
import torch
import math
from typing import Iterable
import time

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_epoch: int, max_norm: float = 0, log: bool = False):
    start_time = time.time()
    model.to(device)
    model.train()
    epoch_loss = 0
    idx = 0
    total_idx = 0
    print(f"\n>>> Epoch #{(epoch+1)}")
    for samples, targets in data_loader:
        idx += 1
        samples = tuple(i.to(device) for i in samples)
        targets = tuple({k:v.to(device) for k,v in anno.items()} for anno in targets)

        loss_dict = model(samples, targets)
        losses = sum(l for l in loss_dict.values())

        if not math.isfinite(losses):
            print("Loss is {}, stopping training".format(losses))
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        epoch_loss += losses
        if idx % 100 == 0:
          print(f"Epoch {epoch+1}/{max_epoch} in Step {idx}/{len(data_loader)}")
          print(f"Total Loss: {epoch_loss/(idx):.4f}, Step Loss: {losses:.4f}")

          elapsed_time = time.time()-start_time
          elapsed_time_str = f"{str(int(elapsed_time//3600)).zfill(2)}H {str(int((elapsed_time%3600)//60)).zfill(2)}M {elapsed_time%60:.2f}S"

          estimaed_time = (time.time()-start_time)*(len(data_loader)/idx)
          estimaed_time_str = f"{str(int(estimaed_time//3600)).zfill(2)}H {str(int((estimaed_time%3600)//60)).zfill(2)}M {estimaed_time%60:.2f}S"

          remain_time = estimaed_time-elapsed_time
          remain_time_str = f"{str(int(remain_time//3600)).zfill(2)}H {str(int((remain_time%3600)//60)).zfill(2)}M {remain_time%60: .2f}S"

          print(f"Time Elapsed: {elapsed_time_str} , Time Estimated: {estimaed_time_str}, Time Remained: {remain_time_str}\n")
    total_idx+=idx
