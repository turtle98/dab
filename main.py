import argparse
import datetime
import json
import random
import time
import multiprocessing
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from dab.data.dataprep import build, collate_fn
import torchvision.models as models
from dab.engine.trainer import train_one_epoch
from dab.engine.arg_parser import get_args_parser

#######################################################
# * function to save our checkpoint when training
#######################################################
def save_ckpt(args, model, optimizer, lr_scheduler, epoch, filename):
    output_dir = Path(args.output_dir)
    if args.output_dir:
        checkpoint_path = output_dir / f'{filename}.pth'
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }, checkpoint_path)

#######################################################
# * main function
#######################################################   
def main(args):
    if args.frozen_weights is not None:
        print("Freeze weights for detector")

    device = torch.device(args.device)
    #######################################################
    # * fix the seed for reproducibility
    #######################################################

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #######################################################
    # * load food dataset 
    #######################################################

    dataset_train = build(args = args)
    num_classes = len(dataset_train.coco.cats)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True) 
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                  collate_fn= collate_fn, num_workers=args.num_workers)
    
    ########################################################
    # *load model, optimizer, criterion
    ########################################################

    model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(num_classes = num_classes)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    
    ########################################################
    # *training
    ########################################################
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, args.epochs)
        lr_scheduler.step()
        save_ckpt(args, model, optimizer, lr_scheduler, epoch, filename='checkpoint_food_model')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'End-to-End food detection model training',
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir += f"/{args.run_name}/"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)