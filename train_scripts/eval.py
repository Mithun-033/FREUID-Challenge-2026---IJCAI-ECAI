from pathlib import Path

import pandas as pd
import torch
from torchvision import transforms
from torchvision.io import decode_image
from tqdm import tqdm

from Model.Config import config, data_config
from Model.model import Model
from data.Dataloaders import dataloader

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=Model(config)
state_dict=torch.load("best_model_optuna_15trials.pt", map_location=device)
missing, unexpected = model.load_state_dict(state_dict, strict=False)

print("Missing:", missing)
print("Unexpected:", unexpected)
conf = data_config()


transform=transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((conf.image_dim, conf.image_dim)),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
image_dir=Path(r"C:\Users\mithu\Desktop\Mithun\FREUID Challenge 2026 - IJCAI-ECAI\public_test\public_test")
image_paths=sorted([
    p for p in image_dir.iterdir()
    if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}
])
results=[]
with torch.no_grad():
    for image_path in tqdm(image_paths, desc="Evaluating", unit="image"):
        image=decode_image(str(image_path))
        image=transform(image).unsqueeze(0).to(device)
        logit=model(image).squeeze()
        score=torch.sigmoid(logit).item()
        results.append({
            "id":image_path.stem,
            "label":score
        })
df=pd.DataFrame(results)
df.to_csv("predictions_4.csv", index=False)
print(f"Saved {len(results)} predictions to predictions_4.csv")
sample = pd.read_csv(r"C:\Users\mithu\Desktop\Mithun\FREUID Challenge 2026 - IJCAI-ECAI\sample_submission.csv")
preds = pd.read_csv("predictions_4.csv")

sample["label"]=sample["label"].astype(float)

sample=sample.set_index("id")
preds=preds.set_index("id")

sample.loc[preds.index,"label"]=preds["label"]

sample.reset_index().to_csv("submission_4.csv", index=False)

#     data = dataloader(
#         train_dir="train/",
#         train_labels="train_labels.csv",
#         val_dir="val/",
#         val_labels="val_labels.csv",
#         config=data_config(),
#     )
#     data.setup()
#     data.prepare_data()
#     train_dataloader = data.train_dataloader()
#     val_dataloader = data.val_dataloader()

#     model.to(device)
#     model.eval()

#     with torch.no_grad():
#         ls = model.block1[0].layerscale.detach().cpu().view(-1)

#         print(ls.min())
#         print(ls.max())
#         print(ls[:20])
#         print(model.block1[0].layerscale.mean())
#         print(model.block2[0].layerscale.mean())
#         print(model.block3[0].layerscale.mean())
#         print(model.block4[0].layerscale.mean())

#         print(model.Linear[2].weight.shape)
#         print(model.Linear[2].weight.mean())
#         print(model.Linear[2].weight.std())
#         print(model.Linear[2].bias)

#         w=model.Linear[2].weight.detach().cpu().flatten()

#         print("abs mean:", w.abs().mean())
#         print("abs max :", w.abs().max())
#         print("first 30:", w[:30])
#         images, labels = next(iter(val_dataloader))
#         images=images[:16].to(device, memory_format=torch.channels_last)

#         print("=" * 80)
#         print("INPUT")
#         print("=" * 80)
#         print(f"mean={images.mean():.6f} std={images.std():.6f} "
#               f"min={images.min():.6f} max={images.max():.6f}")

#         x=model.stem(images)
#         print("\nStem")
#         print(f"shape={tuple(x.shape)}")
#         print(f"mean={x.mean():.6f} std={x.std():.6f} "
#               f"min={x.min():.6f} max={x.max():.6f}")

#         for block in model.block1:
#             x=block(x)

#         print("\nStage 1")
#         print(f"shape={tuple(x.shape)}")
#         print(f"mean={x.mean():.6f} std={x.std():.6f} "
#               f"min={x.min():.6f} max={x.max():.6f}")

#         x=model.down_sample_1(x)

#         print("\nDownsample 1")
#         print(f"shape={tuple(x.shape)}")
#         print(f"mean={x.mean():.6f} std={x.std():.6f} "
#               f"min={x.min():.6f} max={x.max():.6f}")

#         for block in model.block2:
#             x=block(x)

#         print("\nStage 2")
#         print(f"shape={tuple(x.shape)}")
#         print(f"mean={x.mean():.6f} std={x.std():.6f} "
#               f"min={x.min():.6f} max={x.max():.6f}")

#         x=model.down_sample_2(x)

#         print("\nDownsample 2")
#         print(f"shape={tuple(x.shape)}")
#         print(f"mean={x.mean():.6f} std={x.std():.6f} "
#               f"min={x.min():.6f} max={x.max():.6f}")

#         for block in model.block3:
#             x=block(x)

#         print("\nStage 3")
#         print(f"shape={tuple(x.shape)}")
#         print(f"mean={x.mean():.6f} std={x.std():.6f} "
#               f"min={x.min():.6f} max={x.max():.6f}")

#         x=model.down_sample_3(x)

#         print("\nDownsample 3")
#         print(f"shape={tuple(x.shape)}")
#         print(f"mean={x.mean():.6f} std={x.std():.6f} "
#               f"min={x.min():.6f} max={x.max():.6f}")

#         for block in model.block4:
#             x=block(x)

#         print("\nStage 4")
#         print(f"shape={tuple(x.shape)}")
#         print(f"mean={x.mean():.6f} std={x.std():.6f} "
#               f"min={x.min():.6f} max={x.max():.6f}")

#         x=model.global_pool(x)

#         print("\nGlobal Pool")
#         print(f"shape={tuple(x.shape)}")
#         print(f"mean={x.mean():.6f} std={x.std():.6f} "
#               f"min={x.min():.6f} max={x.max():.6f}")

#         features=x.squeeze(-1).squeeze(-1)

#         print("\nFeature Vector")
#         print(f"shape={tuple(features.shape)}")
#         print(f"mean={features.mean():.6f} std={features.std():.6f} "
#               f"min={features.min():.6f} max={features.max():.6f}")

#         hidden=model.Linear[0](features)

#         print("\nLinear 1")
#         print(f"shape={tuple(hidden.shape)}")
#         print(f"mean={hidden.mean():.6f} std={hidden.std():.6f} "
#               f"min={hidden.min():.6f} max={hidden.max():.6f}")

#         hidden=model.Linear[1](hidden)

#         print("\nGELU")
#         print(f"shape={tuple(hidden.shape)}")
#         print(f"mean={hidden.mean():.6f} std={hidden.std():.6f} "
#               f"min={hidden.min():.6f} max={hidden.max():.6f}")
#         print("\n" + "="*80)
#         print("HIDDEN FEATURES")
#         print("="*80)

#         print("Hidden shape:", hidden.shape)

#         print("Across-batch feature std:")
#         print(hidden.std(dim=0).mean().item())

#         print("Per-sample feature std:")
#         print(hidden.std(dim=1).mean().item())

#         print("First sample (first 30 dims):")
#         print(hidden[0, :30].cpu())

#         print("Second sample (first 30 dims):")
#         print(hidden[1, :30].cpu())

#         print("Mean absolute difference between first two samples:")
#         print((hidden[0]-hidden[1]).abs().mean().item())

#         logits=model.Linear[2](hidden)

#         print("\nFinal Logits")
#         print(f"shape={tuple(logits.shape)}")
#         print(f"mean={logits.mean():.6f} std={logits.std():.6f} "
#               f"min={logits.min():.6f} max={logits.max():.6f}")

#         print("\nLogits:")
#         print(logits.squeeze().cpu())