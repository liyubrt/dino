import os
from PIL import Image
import pandas as pd
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

class CustomLargeDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        # load OpenImages data df
        self.oi_data_path = '/data2/jupiter/datasets/OpenImages'
        oi_label_df = pd.read_csv(os.path.join(self.oi_data_path, 'annotations', f'train_600classes.csv'))
        oi_label_df['data'] = 'openimages'
        # load COYO300M data df
        self.coyo_data_path = '/data2/jupiter/datasets/coyo-700m-webdataset'
        coyo_label_df = pd.read_parquet(os.path.join(self.coyo_data_path, 'matches_downloaded', 'label_02p.parquet'))
        coyo_label_df = coyo_label_df[(coyo_label_df.corrupted == False)]
        coyo_label_df['data'] = 'coyo300m'
        # combine two dfs
        self.df = pd.concat([oi_label_df[['data', 'ImageID']], coyo_label_df[['data', 'part1m_dir', 'key']]], ignore_index=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        if row['data'] == 'openimages':
            img_path = os.path.join(self.oi_data_path, 'train', row['ImageID']+'.jpg')
        else:
            img_path = os.path.join(self.coyo_data_path, row['part1m_dir'], f'{str(int(row.key)).zfill(9)}.jpg')
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)