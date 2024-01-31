from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader as TorchDataLoader
from pathlib import Path

class DataLoader:
    def __init__(self, train_dir, val_dir, test_dir):
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.test_dir = Path(test_dir)
        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])
        self.val_test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def create_datasets(self):
        train_dataset = ImageFolder(str(self.train_dir), transform=self.train_transforms)
        val_dataset = ImageFolder(str(self.val_dir), transform=self.val_test_transforms)
        test_dataset = ImageFolder(str(self.test_dir), transform=self.val_test_transforms)
        return train_dataset, val_dataset, test_dataset

    def create_dataloaders(self, batch_size=32):
        train_dataset, val_dataset, test_dataset = self.create_datasets()
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

