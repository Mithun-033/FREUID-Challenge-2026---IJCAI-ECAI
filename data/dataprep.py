import pandas as pd
import os
import shutil

# os.makedirs("val", exist_ok=True)

# df = pd.read_csv("train_labels.csv")

# df_val = pd.DataFrame(columns=df.columns)
# count_1 = count_0 = 0
# remove_idx = []

# for index, row in df.iterrows():
#     if row["label"] == 1 and count_1 < 2143:
#         df_val.loc[len(df_val)] = row
#         count_1 += 1
#         src = src = os.path.join("train", row["image_path"])

#         shutil.move(src, os.path.join("val", os.path.basename(row["image_path"])))
#         remove_idx.append(index)

#     elif row["label"] == 0 and count_0 < 2857:
#         df_val.loc[len(df_val)] = row
#         count_0 += 1
#         src = os.path.join("train", row["image_path"])

#         shutil.move(src, os.path.join("val", os.path.basename(row["image_path"])))
#         remove_idx.append(index)

#     if count_1 >= 2143 and count_0 >= 2857:
#         break

# df.drop(remove_idx, inplace=True)

# df.to_csv("train_labels.csv", index=False)
# df_val.to_csv("val_labels.csv", index=False)

df  = pd.read_csv("val_labels.csv")

df["image_path"] = df["image_path"].str[6:]

df.to_csv("val_labels.csv", index=False)