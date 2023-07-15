import time
import os
import math
import datetime
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets
from torchvision import transforms


class KFoodDataset(datasets.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        super().__init__(root, annFile, transform, target_transform)

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return np.array(Image.open(os.path.join(self.root, path)).convert("RGB"))

    def _load_target(self, id: int):
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        boxes = anns[0]['bbox']
        boxes[2] = boxes[0]+boxes[2]
        boxes[3] = boxes[1]+boxes[3]

        labels = anns[0]['category_id']

        return {'boxes': boxes, 'labels':labels}

    def _rearragne_bbox(self, bbox):
        x1, y1, w, h = bbox
        return x1, y1,x1+w, y1+h


    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        target['boxes'] = torch.Tensor([target['boxes']]).to(torch.int)
        target['labels'] = torch.Tensor([target['labels']]).to(torch.int64)

        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    
def build(args):
    root = Path(args.data_path)
    assert root.exists(), f'provided path {root} does not exist'
    img_folder = root / 'Train'
    ann_file = root /'./food_train_anno.json'
    kfood_dataset = KFoodDataset(
    root = img_folder,
    annFile = ann_file,
    transform = transform,
    )
    return kfood_dataset