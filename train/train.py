import torch.optim as optim
import torch.nn as nn
import torch

from Model.Config import config, data_config
from Model.model import Model
from data.Dataloaders import dataloader

from rich import print, panel
from tqdm import tqdm
import sys
import json


device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(epochs):
    model = Model(config()).to(device).to(torch.channels_last)
    model = torch.compile(model, mode="max-autotune", fullgraph=False).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()  
    
    data = dataloader(train_dir = ..., train_labels = ..., val_dir = ..., val_labels = ..., config = data_config())

    train_dataloader, val_dataloader = data.train_dataloader, data.val_dataloader

    train_loss = []
    val_loss = []

    print(panel.Panel("Starting training ...",style="green"))
    for epoch in range(epochs):
        with tqdm(train_dataloader, desc = "Pretraining", file = sys.stdout) as pbar:
            for img, label in train_dataloader:
                img, label = img.to(device), label.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(img)
                    loss = criterion(output, label)
                    loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                train_loss.append(loss.item())

                ...
    
if __name__ == "__main__":
    train_model(epochs = 2)
