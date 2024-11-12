from PIL import Image
import torch
from torchvision import datasets


def pil_loader(p):
    return Image.open(p).convert("RGB")

class UnlabeledDatasetFolder(datasets.DatasetFolder):
    def __getitem__(self, index):
        path, _ = self.samples[index]  # Ignore the label
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform):
        self.data = datasets.DatasetFolder(data_dir, loader=pil_loader, extensions=["jpg"], transform=transform)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index][0]