import pandas as pd
import os


dir = "results/100epoch_test"

# collect all csvs into a single dataframe by walking through the directory
dfs = []

print(f"Walking through {dir}...")
for root, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith(".csv"):
            # print(f"Reading {file}...")
            df = pd.read_csv(os.path.join(root, file))
            dfs.append(df)
            
# concatenate all dataframes into a single dataframe
combined_df = pd.concat(dfs, ignore_index=True)
print("Combined DataFrame:")
print(combined_df.head())


# plot "val_bacc" over "epoch" for each "model_name" and benchmark with std over "fold"
import matplotlib.pyplot as plt
import seaborn as sns


for benchmark in combined_df["benchmark"].unique():
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_df[combined_df["benchmark"] == benchmark], x="epoch", y="val_bacc", hue="model", errorbar="sd")
    plt.title(f"Validation Balanced Accuracy over Epochs for {benchmark}")
    plt.xlabel("Epoch")
    plt.ylabel("Balanced Accuracy")
    plt.savefig(f"val_bacc_over_epochs_{benchmark}.png")