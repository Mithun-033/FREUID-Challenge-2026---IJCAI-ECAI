# from pathlib import Path

# import pandas as pd
# import torch
# from torchvision import transforms
# from torchvision.io import decode_image
# from tqdm import tqdm

# from Model.Config import config
# from Model.model import Model

# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model=Model(config)

# state_dict=torch.load("model_epoch_10.pt", map_location=device)

# state_dict={
#     k.replace("_orig_mod.", ""):v
#     for k,v in state_dict.items()
# }

# model.load_state_dict(state_dict)
# model.to(device)
# model.eval()

# transform=transforms.Compose([
#     transforms.ConvertImageDtype(torch.float32),
#     transforms.Resize((224,224)),
#     transforms.Normalize(
#         mean=[0.485,0.456,0.406],
#         std=[0.229,0.224,0.225]
#     )
# ])

# image_dir=Path(r"C:\Users\mithu\Desktop\Mithun\FREUID Challenge 2026 - IJCAI-ECAI\the-freuid-challenge-2026-ijcai-ecai\public_test\public_test")

# image_paths=sorted([
#     p for p in image_dir.iterdir()
#     if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}
# ])

# results=[]

# with torch.no_grad():
#     for image_path in tqdm(image_paths, desc="Evaluating", unit="image"):
#         image=decode_image(str(image_path))
#         image=transform(image).unsqueeze(0).to(device)

#         logit=model(image).squeeze()
#         score=torch.sigmoid(logit).item()

#         results.append({
#             "id":image_path.stem,
#             "label":score
#         })

# df=pd.DataFrame(results)
# df.to_csv("predictions_1.csv", index=False)

# print(f"Saved {len(results)} predictions to predictions.csv")

import pandas as pd

sample = pd.read_csv(r"C:\Users\mithu\Desktop\Mithun\FREUID Challenge 2026 - IJCAI-ECAI\the-freuid-challenge-2026-ijcai-ecai\sample_submission.csv")
preds = pd.read_csv("predictions_1.csv")

sample["label"]=sample["label"].astype(float)

sample=sample.set_index("id")
preds=preds.set_index("id")

sample.loc[preds.index,"label"]=preds["label"]

sample.reset_index().to_csv("submission.csv", index=False)