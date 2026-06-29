from torch.utils.data import DataLoader,Dataset
from lightning.pytorch import LightningDataModule
import torchvision.transforms as transforms
from torchvision.io import decode_image
import pandas as pd
import os


class Data(Dataset):
    def __init__(self,Data_dir,labels_csv,is_train):
        super().__init__()

        dataframe=pd.DataFrame(labels_csv)
        self.data_dir=Data_dir

        self.labels=dataframe["label"]
        self.image_paths=dataframe["image_path"]

        train_transform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ColorJitter(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        val_transform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.transform=train_transform if is_train else val_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        img=decode_image(os.path.join(self.data_dir,self.image_paths[idx]))
        img=self.transform(img)

        label=self.labels[idx]

        return img, label
        

class dataloader(LightningDataModule):
    def __init__(self,train_dir,train_labels,val_dir,val_labels,config):
        super().__init__()

        self.train_dir=train_dir
        self.train_labels=train_labels

        self.config=config

        self.val_dir=val_dir
        self.val_labels=val_labels

        self.train=None
        self.val=None

    def prepare_data(self):
        assert self.train_dir is not None, "train dir path not provided"
        assert self.train_labels is not None,  "train labels csv path not provided"

        assert self.config is not None, "config dataclass not provided"

        assert self.val_dir is not None, "val dir path not provided"
        assert self.val_labels is not None,  "val labels csv path not provided"
        
    def setup(self, stage=None):
        self.train=Data(
            self.train_dir,
            self.train_labels,
            is_train=True
        )
        
        self.val=Data(
            self.val_dir,
            self.val_labels,
            is_train=False
        )

    def train_dataloader(self):
        loader = DataLoader(
            self.train,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=True,
            in_order=True
        )
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(
            self.val,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=True,
            in_order=True
        )
        return loader



        

    





