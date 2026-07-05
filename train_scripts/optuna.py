import optuna
import torch
from rich import print, panel

from Model.model import Model
from Model.Config import config, data_config, train_config
from train_scripts.train import train_model, eval_model
from data.Dataloaders import dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(trial, epochs):

    lr = trial.suggest_float("lr", low=1e-4, high=1e-3)
    weight_decay = trial.suggest_float("weight_decay", low=0.01, high=0.1)
    image_dim = trial.suggest_int("image_dim", low=224, high=512, step=16)

    num_block1 = trial.suggest_int("num_block1", low=2, high=5)
    num_block2 = trial.suggest_int("num_block2", low=2, high=5)
    num_block3 = trial.suggest_int("num_block3", low=9, high=12)
    num_block4 = trial.suggest_int("num_block4", low=3, high=6)

    expansion_ratio = trial.suggest_int("expansion_ratio", low=4, high=8)

    dim1 = trial.suggest_categorical("dim1", [84, 96, 108, 120])
    dim2 = dim1 * 2
    dim3 = dim1 * 4
    dim4 = dim1 * 8

    linear_dim = trial.suggest_categorical("linear_dim", [128, 256, 512, 768])

    optuna_model_config = config(
        dim1=dim1,
        dim2=dim2,
        dim3=dim3,
        dim4=dim4,
        num_block1=num_block1,
        num_block2=num_block2,
        num_block3=num_block3,
        num_block4=num_block4,
        expansion_ratio=expansion_ratio,
        linear_dim=linear_dim,
    )

    optuna_train_config = train_config(
        lr=lr,
        weight_decay=weight_decay,
    )

    optuna_data_config = data_config(image_dim=image_dim)

    print(
    panel.Panel(
        f"Trial {trial.number}\n"
        f"lr = {lr:.6f}\n"
        f"weight_decay = {weight_decay:.6f}\n"
        f"image_dim = {image_dim}\n"
        f"num_block1 = {num_block1}\n"
        f"num_block2 = {num_block2}\n"
        f"num_block3 = {num_block3}\n"
        f"num_block4 = {num_block4}\n"
        f"expansion_ratio = {expansion_ratio}\n"
        f"dim1 = {dim1}\n"
        f"linear_dim = {linear_dim}",
        title=f"Trial {trial.number}",
        style="cyan",
        expand=False,
    )
)
    model_dict = train_model(
        epochs=epochs,
        model_con=optuna_model_config,
        data_con=optuna_data_config,
        train_con=optuna_train_config,
        device=device,
        compile=True,
    )

    model = Model(optuna_model_config).to(device)
    state_dict = torch.load(model_dict)
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

    return val_loss


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///optuna_study.db",
        study_name="optuna_study",
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, epochs=1), n_trials=5)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
