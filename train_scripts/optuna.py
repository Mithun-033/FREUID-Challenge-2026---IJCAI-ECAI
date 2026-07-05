import optuna
import torch

from Model.model import Model
from Model.Config import config, data_config, train_config
from train_scripts.train import train_model, eval_model
from data.Dataloaders import dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(trial, epochs):
    model_conf, train_conf, data_conf = config(), train_config(), data_config()

    lr = trial.suggest_float("lr", 1e-4, 1e-3)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
    image_dim = trial.suggest_int("image_dim", 224, 512, 16)

    num_block1 = trial.suggest_int("num_block1", 2, 5)
    num_block2 = trial.suggest_int("num_block2", 2, 5)
    num_block3 = trial.suggest_int("num_block1", 9, 12)
    num_block4 = trial.suggest_int("num_block1", 3, 6)

    expansion_ratio = trial.suggest_int("exp_ratio", 4, 8)
    dim1 = trial.suggest_categorical("dim1", [84, 96, 108, 120])
    linear_dim = trial.suggest_categorical("linear_dim", [128, 256, 512, 768])

    optuna_model_config = model_conf(
        dim1=dim1,
        num_block1=num_block1,
        num_block2=num_block2,
        num_block3=num_block3,
        num_block4=num_block4,
        expansion_ratio=expansion_ratio,
        linear_dim=linear_dim,
    )

    optuna_train_config = train_conf(
        lr=lr,
        weight_decay=weight_decay,
    )

    optuna_data_config = data_conf(image_dim=image_dim)

    model = train_model(
        epochs=epochs,
        model_con=optuna_model_config,
        data_con=optuna_data_config,
        train_con=optuna_train_config,
        device=device,
    )

    state_dict = torch.load(model)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model = torch.load_state_dict(state_dict)
    criterion = torch.nn.BCEWithLogitsLoss()
    data = dataloader(
        train_dir="train/",
        train_labels="train_labels.csv",
        val_dir="val/",
        val_labels="val_labels.csv",
        config=optuna_data_config,
    )
    data.setup()
    data.prepare_data()
    val_dataloader = data.val_dataloader()

    val_loss, acc, pre, rec = eval_model(model, criterion, val_dataloader, device)

    return ...

