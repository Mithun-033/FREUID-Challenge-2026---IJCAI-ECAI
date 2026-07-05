from torch.utils.data import DataLoader,Dataset
from lightning.pytorch import LightningDataModule
import torchvision.transforms as transforms
from torchvision.io import decode_image
import pandas as pd
import os
import torch

class Data(Dataset):
    '''Custom dataset class for loading images and labels from a CSV file.
    Args:
        Data_dir (str): Path to the directory containing the images.
        labels_csv (str): Path to the CSV file containing image paths and labels.
        is_train (bool): Flag indicating whether the dataset is for training or validation.
    '''
    def __init__(self, Data_dir, labels_csv, config, is_train):
        super().__init__()

        dataframe=pd.read_csv(labels_csv)
        self.data_dir=Data_dir

        self.labels=dataframe["label"]
        self.image_paths=dataframe["image_path"]

        train_transform=transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((config.image_dim,config.image_dim)),
            transforms.ColorJitter(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        val_transform=transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((config.image_dim,config.image_dim)),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.transform=train_transform if is_train else val_transform

    def __len__(self):
        '''Returns the total number of samples in the dataset.'''
        return len(self.labels)

    def __getitem__(self,idx):
        '''Returns a single sample from the dataset.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: A tuple containing the image and its corresponding label.
        '''
        img=decode_image(os.path.join(self.data_dir,self.image_paths[idx]))
        img=self.transform(img)

        label=self.labels[idx]

        return img, label
        

class dataloader(LightningDataModule):
    '''
    Custom LightningDataModule for loading training and validation datasets.
    '''
    def __init__(self,train_dir,train_labels,val_dir,val_labels,config):
        '''Initializes the dataloader with training and validation directories, labels, and configuration.
        Args:
            train_dir (str): Path to the training images directory.
            train_labels (str): Path to the training labels CSV file.
            val_dir (str): Path to the validation images directory.
            val_labels (str): Path to the validation labels CSV file.
            config (data_config): Configuration dataclass containing hyperparameters for data loading.
        '''
        super().__init__()

        self.train_dir=train_dir
        self.train_labels=train_labels

        self.config=config

        self.val_dir=val_dir
        self.val_labels=val_labels

        self.train=None
        self.val=None

    def prepare_data(self):
        '''Prepares the data by asserting that the necessary paths and configuration are provided.'''
        assert self.train_dir is not None, "train dir path not provided"
        assert self.train_labels is not None,  "train labels csv path not provided"

        assert self.config is not None, "config dataclass not provided"

        assert self.val_dir is not None, "val dir path not provided"
        assert self.val_labels is not None,  "val labels csv path not provided"
        
    def setup(self, stage=None):
        '''Sets up the training and validation datasets by creating instances of the Data class.
        Args:
            stage (str, optional): Stage of the training process. Defaults to None.
        '''
        self.train=Data(
            self.train_dir,
            self.train_labels,
            self.config,
            is_train=True, 
        )
        
        self.val=Data(
            self.val_dir,
            self.val_labels,
            self.config,
            is_train=False
        )

    def train_dataloader(self):
        '''
        Returns the DataLoader for the training dataset.
        '''
        loader = DataLoader(
            self.train,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=True,
            in_order=False
        )
        return loader
    
    def val_dataloader(self):
        '''
        Returns the DataLoader for the validation dataset.
        '''
        loader = DataLoader(
            self.val,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=True,
            in_order=False
        )
        return loader
    



        

    





