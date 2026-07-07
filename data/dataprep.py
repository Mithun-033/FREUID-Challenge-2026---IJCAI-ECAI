import pandas as pd
import os
import shutil

os.makedirs("val", exist_ok=True)

df = pd.read_csv("train_labels.csv")

df_val = pd.DataFrame(columns=df.columns)
count_1 = count_0 = 0
remove_idx = []

for index, row in df.iterrows():
    if row["label"] == 1 and count_1 < 429:
        df_val.loc[len(df_val)] = row
        count_1 += 1
        src = os.path.join("train", row["image_path"])

        shutil.move(src, os.path.join("val", os.path.basename(row["image_path"][6:])))
        remove_idx.append(index)

    elif row["label"] == 0 and count_0 < 571:
        df_val.loc[len(df_val)] = row
        count_0 += 1
        src = os.path.join("train", row["image_path"])

        shutil.move(src, os.path.join("val", os.path.basename(row["image_path"][6:])))
        remove_idx.append(index)

    if count_1 >= 429 and count_0 >= 571:
        break

df.drop(remove_idx, inplace=True)
df_val["image_path"] = df_val["image_path"].str[6:]
df.to_csv("train_labels.csv", index=False)
df_val.to_csv("val_labels.csv", index=False)

train_dir_count=len([
    f for f in os.listdir("train/train")
    if os.path.isfile(os.path.join("train/train",f))
])

val_dir_count=len([
    f for f in os.listdir("val")
    if os.path.isfile(os.path.join("val",f))
])

train_labels_count=len(pd.read_csv("train_labels.csv"))
val_labels_count=len(pd.read_csv("val_labels.csv"))

print(f"train/train directory: {train_dir_count} files")
print(f"val directory: {val_dir_count} files")
print(f"train_labels.csv: {train_labels_count} rows")
print(f"val_labels.csv: {val_labels_count} rows")





