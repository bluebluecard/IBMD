import os
import io

from PIL import Image
import lmdb

import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import Dataset

from dataset.imagenet import lmdb_loader

def build_lmdb_dataset(
        data_path, pt_path, lmdb_path):
    """
    You can create this dataloader using:
    train_data = _build_lmdb_dataset(traindir, transform=train_transform)
    valid_data = _build_lmdb_dataset(validdir, transform=val_transform)
    """

    root = str(root)
    if root.endswith('/'):
        root = root[:-1]
    pt_path = os.path.join(
        root + '_faster_imagefolder.lmdb.pt')
    lmdb_path = os.path.join(
        root + '_faster_imagefolder.lmdb')
    # root = data_path
    # pt_path = lmdb_path[:-27] + '_faster_imagefolder.lmdb.pt'

    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
        log.info('[Dataset] Loading pt {} and lmdb {}'.format(pt_path, lmdb_path))
        data_set = torch.load(pt_path)
    else:
        data_set = datasets.ImageFolder(
            root, None, None, None)
        torch.save(data_set, pt_path, pickle_protocol=4)
        log.info('[Dataset] Saving pt to {}'.format(pt_path))
        log.info('[Dataset] Building lmdb to {}'.format(lmdb_path))
        env = lmdb.open(lmdb_path, map_size=1e12)
        with env.begin(write=True) as txn:
            for _path, class_index in data_set.imgs:
                with open(_path, 'rb') as f:
                    data = f.read()
                txn.put(_path.encode('ascii'), data)
    data_set.lmdb_data = lmdb.open(
        lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False,
        meminit=False)
    # reset transform and target_transform
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    
    data_set.loader = lambda path: loader(path, data_set.lmdb_data)

    return data_set
