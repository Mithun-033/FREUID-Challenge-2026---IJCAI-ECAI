import torch.optim as optim
import torch.nn as nn
import torch
import torchmetrics 
import torch._dynamo as dynamo

from Model.Config import config, data_config, train_config
from Model.model import Model
from data.Dataloaders import dataloader

import argparse
from rich import print, panel
from tqdm import tqdm
import sys
import json


def eval_model(model, criterion, val_dataloader, device):
    model.eval()

    val_loss_accum = 0.0
    acc_metric = torchmetrics.Accuracy(task="binary").to(device)
    prec_metric = torchmetrics.Precision(task="binary").to(device)
    rec_metric = torchmetrics.Recall(task="binary").to(device)
    acc_metric.reset()
    prec_metric.reset()
    rec_metric.reset()
    with torch.no_grad():
        for x_val, y_val in val_dataloader:
            x_val = x_val.to(device, memory_format = torch.channels_last)
            y_val = y_val.to(device).unsqueeze(1).float()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(x_val)
                loss = criterion(output, y_val)
            val_loss_accum += loss.item()
            acc_metric.update(output, y_val.int())
            prec_metric.update(output, y_val.int())
            rec_metric.update(output, y_val.int())

    avg_val_loss = val_loss_accum / len(val_dataloader)
    avg_accuracy = acc_metric.compute().item()
    avg_precision = prec_metric.compute().item()
    avg_recall = rec_metric.compute().item()

    return avg_val_loss, avg_accuracy, avg_precision, avg_recall


def train_model(epochs, model_con, data_con, train_con, device, compile = False):
    
    model = Model(model_con).to(device, memory_format = torch.channels_last)
    if compile:
        model = torch.compile(model).to(device)

    params = 0
    for param in model.parameters():
        params += param.numel()

    print(panel.Panel(f"Total number of parameters: {params:_}", style="blue"))

    optimizer = optim.AdamW(model.parameters(), lr=train_con.lr, weight_decay=train_con.weight_decay, fused = True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs * (68352) / data_config.batch_size, eta_min = train_con.min_lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([(40_005 - 571) / (29_347 - 429)],device = device))

    train_loss = []
    val_loss = []
    accuracy_list = []
    precision_list = []
    recall_list = []

    step = 0
    loss_acum = 0.0

    print(panel.Panel("Starting training ...",style="green"))
    for epoch in range(epochs):
        data = dataloader(train_dir = "train/", train_labels = "train_labels.csv", val_dir = "val/", val_labels = "val_labels.csv", config = data_con)
        data.setup()
        data.prepare_data()
        train_dataloader, val_dataloader = data.train_dataloader(), data.val_dataloader()
        with tqdm(train_dataloader, desc = "Pretraining", file = sys.stdout) as pbar:
            for img, label in pbar:
                img, label = img.to(device, memory_format = torch.channels_last), label.to(device).unsqueeze(1).float()
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(img)
                    loss = criterion(output, label)
    
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none = True)
                
                step += 1
                loss_acum += loss.item()
                
                if step % 10 == 0:
                    avg_loss = loss_acum / 10
                    train_loss.append(avg_loss)
                    pbar.set_postfix({"loss": avg_loss})
                    loss_acum = 0.0

                if step % 250 == 0:
                    avg_val_loss, avg_accuracy, avg_precision, avg_recall = eval_model(model, criterion, val_dataloader, device)

                    val_loss.append(avg_val_loss)
                    accuracy_list.append(avg_accuracy)
                    precision_list.append(avg_precision)
                    recall_list.append(avg_recall)

                    model.train()

            with open("train_loss.json", "w") as f:
                json.dump(train_loss, f)
            with open("val_loss.json", "w") as f:
                json.dump(val_loss, f)
            with open("accuracy.json", "w") as f:
                json.dump(accuracy_list, f)
            with open("precision.json", "w") as f:
                json.dump(precision_list, f)
            with open("recall.json", "w") as f:
                json.dump(recall_list, f)

            torch.save(model.state_dict(), "model.pt")

    return model.state_dict()

                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    model_con, data_con, train_con = config(), data_config(), train_config()
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train_model(epochs = args.epochs, model_con = model_con,data_con = data_con, train_con = train_con, device = device, compile = True)