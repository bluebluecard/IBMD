# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import io
import random
from tqdm.auto import tqdm
from pathlib import Path

from PIL import Image
import lmdb

import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, Subset

from ipdb import set_trace as debug

def lmdb_loader(path, lmdb_data):
    # In-memory binary streams
    with lmdb_data.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(path.encode())
    img = Image.open(io.BytesIO(bytedata))
    return img.convert('RGB')


def lmdb_loader_root(path, lmdb_data, root):
    # In-memory binary streams
    # print(f"path = {path}")
    # print(f"root = {root}")
    path_from_root = os.path.join(root, path)
    # print(f"path_from_root = {path_from_root}")
    
    with lmdb_data.begin(write=False, buffers=True) as txn:
        bytedata = txn.get(path_from_root.encode())
    img = Image.open(io.BytesIO(bytedata))
    return img.convert('RGB')


def _build_lmdb_dataset(
        root, log, transform=None, target_transform=None,
        loader=lmdb_loader, lmdb_folder_path="/path-to-dataset"):
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
    
    # def loader_fn(path):
    #     return loader(path, data_set.lmdb_data)
    # data_set.loader = loader_fn
    data_set.loader = lambda path: loader(path, data_set.lmdb_data)

    return data_set

def _build_lmdb_dataset_lmdb_folder_path(root, transform=None, target_transform=None, loader=lmdb_loader,
                                         lmdb_folder_path="/mnt/ar_hdd/ sel/imagenet_lmdb"):
    """
    You can create this dataloader using:
    train_data = _build_lmdb_dataset(traindir, transform=train_transform)
    valid_data = _build_lmdb_dataset(validdir, transform=val_transform)
    """

    root = str(root)
    if root.endswith("/"):
        root = root[:-1]
    print(f"root = {root}")
    
    split = root.split("/")[-1]
    
    # path_to_save = "/extra_disk_1/ sel/imagenet_lmdb"
    path_to_save = lmdb_folder_path
    os.makedirs(path_to_save, exist_ok=True)
    path_to_save_split = os.path.join(path_to_save, split)
    os.makedirs(path_to_save_split, exist_ok=True)
    
    pt_path = os.path.join(path_to_save_split, "faster_imagefolder.lmdb.pt")
    lmdb_path = os.path.join(path_to_save_split, "faster_imagefolder.lmdb")
    print(f"pt_path = {pt_path}, lmdb_path = {lmdb_path}")

    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
        # log.info('[Dataset] Loading pt {} and lmdb {}'.format(pt_path, lmdb_path))
        data_set = torch.load(pt_path)
    else:
        data_set = datasets.ImageFolder(root, None, None, None)
        torch.save(data_set, pt_path, pickle_protocol=4)
        # log.info('[Dataset] Saving pt to {}'.format(pt_path))
        # log.info('[Dataset] Building lmdb to {}'.format(lmdb_path))
        map_size=int(1e12)
        print(f"map size = {map_size}")
        env = lmdb.open(lmdb_path, map_size=map_size)
        with env.begin(write=True) as txn:
            for _path, class_index in tqdm(data_set.imgs):
                with open(_path, "rb") as f:
                    data = f.read()
                txn.put(_path.encode("ascii"), data)
    print(f"open {lmdb_path}")
    data_set.lmdb_data = lmdb.open(lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
    # reset transform and target_transform
    print(f"opened!")
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    data_set.loader = lambda path: loader(path, data_set.lmdb_data)

    return data_set

def _build_lmdb_dataset_lmdb_folder_path_val10k(root, transform=None, target_transform=None, loader=lmdb_loader_root,
                                                lmdb_folder_path="/mnt/ar_hdd/ sel/imagenet_lmdb"):
    """
    You can create this dataloader using:
    train_data = _build_lmdb_dataset(traindir, transform=train_transform)
    valid_data = _build_lmdb_dataset(validdir, transform=val_transform)
    """

    root = str(root)
    if root.endswith("/"):
        root = root[:-1]
    print(f"root = {root}")
    
    split = root.split("/")[-1]
    
    # path_to_save = "/extra_disk_1/ sel/imagenet_lmdb"
    path_to_save = lmdb_folder_path
    os.makedirs(path_to_save, exist_ok=True)
    path_to_save_split = os.path.join(path_to_save, split)
    os.makedirs(path_to_save_split, exist_ok=True)
    
    pt_path = os.path.join(path_to_save_split, "faster_imagefolder.lmdb.pt")
    lmdb_path = os.path.join(path_to_save_split, "faster_imagefolder.lmdb")
    print(f"pt_path = {pt_path}, lmdb_path = {lmdb_path}")

    if os.path.isfile(pt_path) and os.path.isdir(lmdb_path):
        # log.info('[Dataset] Loading pt {} and lmdb {}'.format(pt_path, lmdb_path))
        data_set = torch.load(pt_path)
    else:
        data_set = datasets.ImageFolder(root, None, None, None)
        torch.save(data_set, pt_path, pickle_protocol=4)
        # log.info('[Dataset] Saving pt to {}'.format(pt_path))
        # log.info('[Dataset] Building lmdb to {}'.format(lmdb_path))
        map_size=int(1e12)
        print(f"map size = {map_size}")
        env = lmdb.open(lmdb_path, map_size=map_size)
        with env.begin(write=True) as txn:
            for _path, class_index in tqdm(data_set.imgs):
                with open(_path, "rb") as f:
                    data = f.read()
                txn.put(_path.encode("ascii"), data)
    print(f"open {lmdb_path}")
    data_set.lmdb_data = lmdb.open(lmdb_path, readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
    # reset transform and target_transform
    print(f"opened!")
    data_set.samples = data_set.imgs
    data_set.transform = transform
    data_set.target_transform = target_transform
    data_set.loader = lambda path: loader(path, data_set.lmdb_data, root)

    return data_set

def build_train_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Lambda(lambda t: (t * 2) - 1) # [0,1] --> [-1, 1]
        transforms.Normalize(0.5,0.5),
    ])

def build_test_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Lambda(lambda t: (t * 2) - 1) # [0,1] --> [-1, 1]
        transforms.Normalize(0.5,0.5),
    ])


class PairedImageTransform:
    def __init__(self, train):
        self.train = train
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(0.5, 0.5)

    def _transform_image(self, image, flip):
        if flip:
            image = TF.hflip(image)
        image = self.to_tensor(image)
        if image.shape != (3, 640, 360):
            raise RuntimeError(
                f"Expected paired raw tensor shape (3, 640, 360), got {tuple(image.shape)}"
            )
        return self.normalize(image)

    def __call__(self, clean_img, corrupt_img):
        flip = self.train and random.random() < 0.5
        return self._transform_image(clean_img, flip), self._transform_image(corrupt_img, flip)


class PairedImageFolderDataset(Dataset):
    def __init__(self, clean_root, corrupt_root, corrupt_subdir, seq_name=None, transform=None):
        super().__init__()
        self.clean_root = Path(clean_root)
        self.corrupt_root = Path(corrupt_root)
        self.corrupt_subdir = corrupt_subdir
        self.seq_name = seq_name
        self.transform = transform
        self.sequence_names = self._build_sequence_names()
        self.samples = self._build_samples()

    def _build_sequence_names(self):
        if self.seq_name is not None:
            return [self.seq_name]

        clean_hr_root = self.clean_root / "HR"
        corrupt_hr_root = self.corrupt_root / "HR"
        if not clean_hr_root.is_dir():
            raise RuntimeError(f"Missing clean HR directory: {clean_hr_root}")
        if not corrupt_hr_root.is_dir():
            raise RuntimeError(f"Missing corrupt HR directory: {corrupt_hr_root}")

        clean_sequences = sorted(path.name for path in clean_hr_root.iterdir() if path.is_dir())
        if not clean_sequences:
            raise RuntimeError(f"No clean sequence directories found under {clean_hr_root}")

        corrupt_sequences = sorted(path.name for path in corrupt_hr_root.iterdir() if path.is_dir())
        if not corrupt_sequences:
            raise RuntimeError(f"No corrupt sequence directories found under {corrupt_hr_root}")

        missing_corrupt_sequences = sorted(set(clean_sequences) - set(corrupt_sequences))
        extra_corrupt_sequences = sorted(set(corrupt_sequences) - set(clean_sequences))
        if missing_corrupt_sequences or extra_corrupt_sequences:
            error_lines = [
                "Clean/corrupt sequence mismatch detected.",
                f"clean_root={self.clean_root}",
                f"corrupt_root={self.corrupt_root}",
                f"corrupt_subdir={self.corrupt_subdir}",
            ]
            if missing_corrupt_sequences:
                error_lines.append(f"Missing corrupt sequence: {missing_corrupt_sequences[0]}")
                error_lines.append(f"Total missing corrupt sequences: {len(missing_corrupt_sequences)}")
            if extra_corrupt_sequences:
                error_lines.append(f"Extra corrupt sequence: {extra_corrupt_sequences[0]}")
                error_lines.append(f"Total extra corrupt sequences: {len(extra_corrupt_sequences)}")
            raise RuntimeError(" | ".join(error_lines))

        return clean_sequences

    def _build_sequence_samples(self, seq_name):
        clean_seq_root = self.clean_root / "HR" / seq_name
        corrupt_seq_root = self.corrupt_root / "HR" / seq_name / self.corrupt_subdir

        clean_files = sorted(clean_seq_root.glob("*.png"))
        if not clean_files:
            raise RuntimeError(f"No clean PNG files found under {clean_seq_root}")

        corrupt_files = sorted(corrupt_seq_root.glob("*.png"))
        if not corrupt_files:
            raise RuntimeError(
                f"No corrupt PNG files found under {corrupt_seq_root}"
            )

        clean_map = {
            path.name: path
            for path in clean_files
        }
        corrupt_map = {
            path.name: path
            for path in corrupt_files
        }

        missing_corrupt = sorted(set(clean_map) - set(corrupt_map))
        extra_corrupt = sorted(set(corrupt_map) - set(clean_map))

        if missing_corrupt or extra_corrupt:
            error_lines = [
                "Clean/corrupt dataset mismatch detected.",
                f"clean_root={self.clean_root}",
                f"corrupt_root={self.corrupt_root}",
                f"corrupt_subdir={self.corrupt_subdir}",
            ]
            if missing_corrupt:
                file_name = missing_corrupt[0]
                error_lines.append(f"Missing corrupt example: sequence={seq_name}, file={file_name}")
                error_lines.append(f"Total missing corrupt files: {len(missing_corrupt)}")
            if extra_corrupt:
                file_name = extra_corrupt[0]
                error_lines.append(f"Extra corrupt example: sequence={seq_name}, file={file_name}")
                error_lines.append(f"Total extra corrupt files: {len(extra_corrupt)}")
            raise RuntimeError(" | ".join(error_lines))

        return [
            (clean_map[key], corrupt_map[key])
            for key in sorted(clean_map)
        ]

    def _build_samples(self):
        samples = []
        for seq_name in self.sequence_names:
            samples.extend(self._build_sequence_samples(seq_name))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        clean_path, corrupt_path = self.samples[index]
        clean_img = Image.open(clean_path).convert("RGB")
        corrupt_img = Image.open(corrupt_path).convert("RGB")

        if self.transform is not None:
            clean_img, corrupt_img = self.transform(clean_img, corrupt_img)

        return clean_img, corrupt_img

def build_lmdb_dataset(opt, log, train, transform=None):
    """ resize -> crop -> to_tensor -> norm(-1,1) """
    fn = opt.dataset_dir / ('train' if train else 'val')

    if transform is None:
        build_transform = build_train_transform if train else build_test_transform
        transform = build_transform(opt.image_size)

    dataset = _build_lmdb_dataset(fn, log, transform=transform)
    log.info(f"[Dataset] Built Imagenet dataset {fn=}, size={len(dataset)}!")
    return dataset


def build_paired_dataset(opt, log, train, transform=None):
    if train:
        split = "train"
    else:
        split = "train" if getattr(opt, "val_on_train", False) else "val"
    clean_root = opt.clean_dataset_dir / split
    corrupt_root = opt.corrupt_dataset_dir / split

    if transform is None:
        transform = PairedImageTransform(train=train)

    dataset = PairedImageFolderDataset(
        clean_root=clean_root,
        corrupt_root=corrupt_root,
        corrupt_subdir=opt.corrupt_subdir,
        seq_name=opt.seq_name,
        transform=transform,
    )
    log.info(
        f"[Dataset] Built paired dataset split={split}, clean_root={clean_root}, "
        f"corrupt_root={corrupt_root}, corrupt_subdir={opt.corrupt_subdir}, "
        f"sequence_filter={opt.seq_name or 'all'}, sequences={len(dataset.sequence_names)}, size={len(dataset)}!"
    )
    return dataset


def maybe_limit_dataset(opt, log, dataset, train):
    num_images = getattr(opt, "num_images", None)
    if num_images is None:
        return dataset

    if num_images <= 0:
        raise ValueError(f"num_images must be > 0 when provided, got {num_images}")

    limited_size = min(len(dataset), num_images)
    subset = Subset(dataset, range(limited_size))
    split = "train" if train else "val"
    log.info(
        f"[Dataset] Limited {split} dataset to first {limited_size}/{len(dataset)} samples "
        f"for debugging via num_images={num_images}."
    )
    return subset


def build_dataset(opt, log, train, transform=None):
    dataset_mode = getattr(opt, "dataset_mode", "lmdb")
    if dataset_mode == "paired":
        dataset = build_paired_dataset(opt, log, train, transform=transform)
    else:
        dataset = build_lmdb_dataset(opt, log, train, transform=transform)
    return maybe_limit_dataset(opt, log, dataset, train)

def build_lmdb_dataset_lmdb_folder_path(opt, log, train, transform=None, 
                                        lmdb_folder_path="/mnt/ar_hdd/ sel/imagenet_lmdb"):
    """ resize -> crop -> to_tensor -> norm(-1,1) """
    fn = opt.dataset_dir / ('train' if train else 'val')

    if transform is None:
        build_transform = build_train_transform if train else build_test_transform
        transform = build_transform(opt.image_size)

    dataset = _build_lmdb_dataset_lmdb_folder_path(fn, transform=transform,
                                                   lmdb_folder_path=lmdb_folder_path)
    log.info(f"[Dataset] Built Imagenet dataset {fn=}, size={len(dataset)}!")
    return dataset

def readlines(fn):
    file = open(fn, "r").readlines()
    return [line.strip('\n\r') for line in file]

def build_lmdb_dataset_val10k(opt, log, transform=None):

    fn_10k = readlines(f"dataset/val_faster_imagefolder_10k_fn.txt")
    label_10k = readlines(f"dataset/val_faster_imagefolder_10k_label.txt")

    if transform is None: transform = build_test_transform(opt.image_size)
    dataset = _build_lmdb_dataset(opt.dataset_dir / 'val', log, transform=transform)
    dataset.samples = [(fn, int(label)) for fn, label in zip(fn_10k, label_10k)]

    assert len(dataset) == 10_000
    log.info(f"[Dataset] Built Imagenet val10k, size={len(dataset)}!")
    return dataset

def build_lmdb_dataset_val10k_lmdb_folder_path(opt, log, transform=None,
                                               lmdb_folder_path="/mnt/ar_hdd/ sel/imagenet_lmdb"):

    fn_10k = readlines(f"dataset/val_faster_imagefolder_10k_fn_dbim.txt")
    label_10k = readlines(f"dataset/val_faster_imagefolder_10k_label.txt")

    if transform is None: transform = build_test_transform(opt.image_size)
    # dataset = _build_lmdb_dataset(opt.dataset_dir / 'val', log, transform=transform)
    
    dataset = _build_lmdb_dataset_lmdb_folder_path_val10k(opt.dataset_dir / 'val', transform=transform,
                                                          lmdb_folder_path=lmdb_folder_path)
    dataset.samples = [(fn, int(label)) for fn, label in zip(fn_10k, label_10k)]

    assert len(dataset) == 10_000
    log.info(f"[Dataset] Built Imagenet val10k, size={len(dataset)}!")
    return dataset

class InpaintingVal10kSubset(Dataset):
    def __init__(self, opt, log, mask):
        super(InpaintingVal10kSubset, self).__init__()

        assert mask in ["center", "freeform1020", "freeform2030"]
        self.mask_type = mask
        self.dataset = build_lmdb_dataset_val10k(opt, log)

        from corruption.inpaint import get_center_mask, load_freeform_masks
        if self.mask_type == "center":
            self.mask = get_center_mask([opt.image_size, opt.image_size]) # [1,256,256]
        else:
            self.masks = load_freeform_masks(mask)[:,0,...] # [10000, 256, 256]
            assert len(self.dataset) == len(self.masks)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        mask = self.mask if self.mask_type == "center" else self.masks[[index]]
        return *self.dataset[index], mask
    
class InpaintingVal10kSubset_LMDB_folder_path(Dataset):
    def __init__(self, opt, log, mask):
        super(InpaintingVal10kSubset_LMDB_folder_path, self).__init__()

        assert mask in ["center", "freeform1020", "freeform2030"]
        self.mask_type = mask
        self.dataset = build_lmdb_dataset_val10k_lmdb_folder_path(opt, log, 
                                                                  lmdb_folder_path=opt.lmdb_folder_path)

        from corruption.inpaint import get_center_mask, load_freeform_masks
        if self.mask_type == "center":
            self.mask = get_center_mask([opt.image_size, opt.image_size]) # [1,256,256]
        else:
            self.masks = load_freeform_masks(mask)[:,0,...] # [10000, 256, 256]
            assert len(self.dataset) == len(self.masks)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        mask = self.mask if self.mask_type == "center" else self.masks[[index]]
        return *self.dataset[index], mask
