import optuna
import torch
from rich import print, panel
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_curve, auc

from Model.model import Model
from Model.Config import config, data_config, train_config
from train_scripts.train import train, eval_model
from data.Dataloaders import dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def compute_freuid(model, device, val_loader, target_bpcer=0.01):
    """
    labels: 1 = fraudulent document (attack), 0 = real document (bona fide)
    model outputs a single raw logit per sample representing P(fraudulent)
    """
    model.eval()
    all_scores, all_labels = [], []

    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        logits = model(inputs).squeeze(-1)
        probs = torch.sigmoid(logits) 

        all_scores.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    audet = 1 - auc(fpr, tpr)

    apcer_at_target = float(np.interp(target_bpcer, fpr, fnr))

    g_audet = 1 - audet
    g_apcer = 1 - apcer_at_target
    denom = g_audet + g_apcer
    freuid = 1.0 if denom == 0 else 1 - 2 * g_audet * g_apcer / denom

    return {
        "freuid": freuid,
        "g_audet": g_audet,
        "g_apcer": g_apcer,
    }

global_best_freuid = float("inf")

def objective(trial, epochs):
    lr = trial.suggest_float("lr", low=1e-4, high=5e-3, log = True)
    weight_decay = trial.suggest_float("weight_decay", low=0.01, high=0.1, log = True)
    image_dim = trial.suggest_int("image_dim", low=224, high=512, step=16)

    num_block1 = trial.suggest_int("num_block1", low=2, high=5)
    num_block2 = trial.suggest_int("num_block2", low=2, high=5)
    num_block3 = trial.suggest_int("num_block3", low=9, high=12)
    num_block4 = trial.suggest_int("num_block4", low=3, high=6)

    expansion_ratio = trial.suggest_int("expansion_ratio", low=2, high=8, step = 2)

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
    raw_model = Model(optuna_model_config).to(device, memory_format = torch.channels_last)
    model = torch.compile(raw_model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=optuna_train_config.lr, weight_decay=optuna_train_config.weight_decay, fused = True)
    criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([(40_005 - 571) / (29_347 - 429)],device = device))
    
    params = 0
    for param in model.parameters():
        params += param.numel()

    print(panel.Panel(f"Total number of parameters: {params:_}", style="blue"))

    data = dataloader(
    train_dir="train/",
    train_labels="train_labels.csv",
    val_dir="val/",
    val_labels="val_labels.csv",
    config=optuna_data_config,
)
    data.setup()
    data.prepare_data()

    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs * len(train_dataloader), eta_min = optuna_train_config.min_lr)
    global global_best_freuid
    for epoch in range(epochs):
        train(
            model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataloader=train_dataloader,
            device=device,
        )

        freuid_metrics = compute_freuid(
            model,
            device,
            val_dataloader,
            target_bpcer=0.01,
        )

        freuid = freuid_metrics["freuid"]

        if freuid < global_best_freuid:
            global_best_freuid = freuid
            print(panel.Panel(f"New best FREUID: {freuid:.6f} at epoch {epoch + 1}", style="green"))
            torch.save(
                raw_model.state_dict(),
                "best_model_.pt",
            )
        
        trial.report(freuid, epoch)
        if trial.should_prune():
            val_loss, acc, rec, pre = eval_model(
                model,
                criterion,
                val_dataloader,
                device,
            )
            trial.set_user_attr("freuid", freuid)
            trial.set_user_attr("val_loss", val_loss)
            trial.set_user_attr("accuracy", acc)
            trial.set_user_attr("recall", rec)
            trial.set_user_attr("precision", pre)
            trial.set_user_attr("g_audet", freuid_metrics["g_audet"])
            trial.set_user_attr("g_apcer", freuid_metrics["g_apcer"])
            trial.set_user_attr("last_epoch", epoch + 1)

            raise optuna.TrialPruned()

    val_loss, acc, rec, pre = eval_model(
        model,
        criterion,
        val_dataloader,
        device,
    )
    trial.set_user_attr("val_loss", val_loss)
    trial.set_user_attr("accuracy", acc)
    trial.set_user_attr("recall", rec)
    trial.set_user_attr("precision", pre)
    trial.set_user_attr("g_audet", freuid_metrics["g_audet"])
    trial.set_user_attr("g_apcer", freuid_metrics["g_apcer"])
    trial.set_user_attr("last_epoch", epochs)

    return freuid

if __name__ == "__main__":

    pruner = optuna.pruners.HyperbandPruner(
        min_resource = 3,
        max_resource = "auto",
        reduction_factor = 2
    )

    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///optuna_study.db",
        study_name="optuna_study",
        pruner=pruner,
        load_if_exists=False
    )
    study.optimize(lambda trial: objective(trial, epochs=30), n_trials=100)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
