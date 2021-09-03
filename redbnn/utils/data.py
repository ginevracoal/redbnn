import os
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision import datasets, transforms

from redbnn.utils.seeding import set_seed

class TransformDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)

def transform_data(train_set, val_set, test_set, img_size):

    set_seed(0)

    stats = [0.,0.,0.],[1.,1.,1.]

    train_set = TransformDataset(train_set, transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(img_size, padding=None),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats, inplace=True),

        ]))

    val_set = TransformDataset(val_set, transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(*stats, inplace=True)
        ]))

    test_set = TransformDataset(test_set, transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(*stats, inplace=True)
        ]))

    return train_set, val_set, test_set

def load_data(dataset_name, data_dir, phases=['train','val','test'], batch_size=64, subset_size=None, num_workers=0):
    """
    Build a dictionary containing training, validation and test dataloaders from the chosen dataset.
    """
    set_seed(0)
    
    if dataset_name=="imagenette":

        num_classes = 10
        img_size = 224

        transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        train_set = datasets.ImageFolder(data_dir+"/train", transform=transform)
        test_set = datasets.ImageFolder(data_dir+"/test", transform=transform)

        if subset_size:
            train_set = torch.utils.data.Subset(train_set, np.random.choice(len(train_set), subset_size, replace=False))
            test_set = torch.utils.data.Subset(test_set, np.random.choice(len(test_set), subset_size, replace=False))

        val_size = int(0.1 * len(train_set))
        train_size = len(train_set) - val_size
        train_set, val_set = random_split(train_set, [train_size, val_size])
        
        train_set, val_set, test_set = transform_data(train_set, val_set, test_set, img_size)

    else:
        raise NotImplementedError
    
    print("\nimg shape =", next(iter(train_set))[0].shape, "\tnum_classes =", num_classes, end="\t")

    datasets_dict = {'train':train_set, 'val':val_set, 'test':test_set}
    dataloaders_dict = {}

    for phase in phases:
        dataloaders_dict[phase] = DataLoader(dataset=datasets_dict[phase], batch_size=batch_size, shuffle=False)
        print(phase, "dataset length =", len(datasets_dict[phase]), end="\t")

    print()
    return dataloaders_dict, img_size, num_classes