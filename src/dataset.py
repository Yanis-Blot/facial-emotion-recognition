from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from config import TRAIN_DIR, TEST_DIR, BATCH_SIZE, device

# -- Transforms --
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((48,48)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# required transforms
transform_min = transforms.Compose([
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

#-- Load Dataset
def load_datasets(augmented=True):
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform_min)
    test_dataset  = datasets.ImageFolder(root=TEST_DIR, transform=transform_min)

    if augmented:
        train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)

    # Return class names
    class_names = [c for c in train_dataset.classes]

    return train_dataset, test_dataset, class_names

# -- DataLoaders --
def get_dataloader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0, pin_memory = True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)