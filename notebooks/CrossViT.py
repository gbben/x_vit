# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # hey bro have you heard of crossvit?

# %load_ext autoreload
# %autoreload 2

# +
import matplotlib.pyplot as plt
import pandas as pd
import schedulefree
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from datasets import Dataset
from pathlib import Path
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import Trainer, TrainingArguments
from tqdm import tqdm

from constants import *
from weigh_protein.crossvit import GPTCrossViTDualBranch, CrossViTDualBranch
from weigh_protein.utils import formatted_now


# -

def prepare_dataloader(dataset, batch_size, split="train"):
    def collate_fn(batch):
        # Stack RGB and depth images and convert labels to tensors
        rgb_images = torch.stack([item["rgb"] for item in batch])
        depth_images = torch.stack([item["depth"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
        return rgb_images, depth_images, labels
    
    return DataLoader(dataset[split], batch_size=batch_size, shuffle=(split == "train"), collate_fn=collate_fn)


def train(model, train_loader, optimizer, epoch, num_epochs, device):
    model.train()
    if hasattr(optimizer, "train"):  # if using an optimizer from schedulefree
        optimizer.train()
    train_loss = 0.0
    for images_1, images_2, target in tqdm(
        train_dataloader, desc=f"Training Epoch {epoch}/{num_epochs}", ncols=120
    ):
        # Move data to the device
        images_1, images_2, target = images_1.to(device), images_2.to(device), target.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images_1, images_2).squeeze()  # Squeeze to match labels shape

        # Compute loss (MSE for regression)
        loss = F.mse_loss(outputs, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
            
        train_loss += loss.item()
        
    # Calculate average training loss
    avg_train_loss = train_loss / len(train_dataloader)

    #print(f"Epoch [{epoch}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")
        
    # wandb.log({"training loss": avg_train_loss}, step=epoch)
    return avg_train_loss


def validate(model, val_dataloader, optimizer, epoch, num_epochs, device):
    model.eval()
    if hasattr(optimizer, "eval"):  # if using an optimizer from schedulefree
        optimizer.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images_1, images_2, target in tqdm(val_dataloader, desc=f"Validation Epoch {epoch}/{num_epochs}", ncols=120):
            images_1, images_2, target = images_1.to(device), images_2.to(device), target.to(device)
            
            # Forward pass
            outputs = model(images_1, images_2).squeeze()
            
            # Compute loss
            loss = F.mse_loss(outputs, target)
            val_loss += loss.item()
    
    # Calculate average validation loss
    avg_val_loss = val_loss / len(val_dataloader)

    # wandb.log({"validation loss": avg_val_loss}, step=epoch)
    return avg_val_loss


def run_training(model, train_dataloader, val_dataloader, num_epochs=5, learning_rate=1e-4, device="mps"):
    # run = wandb.init(
    #     project="cows",
    #     config={
    #         "learning_rate": learning_rate,
    #         "architecture": "Dual branch CrossViT",
    #         "dataset": "CID",
    #         "epochs": num_epochs,
    #         "branch_base_model": model.branch_model_name,
    #     }
    # )
    # run.watch(model)
    
    # Move model to the specified device
    model = model.to(device)
    
    # Set up optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    
    #optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=learning_rate, warmup_steps=4)

    # Training loop
    for epoch in range(1, num_epochs+1):
        train_loss = train(model, train_dataloader, optimizer, epoch, num_epochs, device)
        val_loss = validate(model, val_dataloader, optimizer, epoch, num_epochs, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch}/{num_epochs}] - Training loss {train_loss:.4f}. Validation loss {val_loss:.4f}. Learning rate {lr:.3f}")
        # wandb.log({"learning_rate": lr}, step=epoch)
        
    run_output_dir = output_dir / formatted_now()
    run_output_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), run_output_dir / "model.pt")
    # run.finish()


# # Create dataset

cid_ds = Dataset.from_parquet(str(cid_dir / "all_angles_regression_ds.pq"))

df = cid_ds.to_pandas()
df["cow_id"] = (df.index // 4) + 1
df = df.groupby("cow_id").head(2).reset_index(drop=True)
df = df.assign(
    image1=df.groupby("cow_id")["image"].transform("first"),
    image2=df.groupby("cow_id")["image"].transform("last")
).drop(columns=["image"])
df = df.drop_duplicates(subset="cow_id").reset_index(drop=True).drop(columns="cow_id")
df.rename(columns={"image1": "rgb", "image2": "depth", "weight_in_kg": "label"}, inplace=True)
df["rgb"] = df["rgb"].map(lambda x: Image.open(x["path"]))
df["depth"] = df["depth"].map(lambda x: Image.open(x["path"]))
df

ds = Dataset.from_dict(df[["rgb", "depth", "label"]].to_dict(orient="list"))
ds[0]

ds[0]["rgb"]

# ## Save

ds.to_parquet(data_dir / "crossvit_ds.pq")

# ## Load

ds = Dataset.from_parquet(str(data_dir / "crossvit_ds.pq"))

# # Preprocess dataset

# +
# Get max value for normalisation
max_value = max(ds["label"])

# Define a transform to convert PIL images to tensors
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def preprocess(example):
    example["rgb"] = transform(example["rgb"])
    example["depth"] = transform(example["depth"])
    example["label"] = example["label"] / max_value  # Normalize values
    return example

# Apply the preprocessing with normalization
ds = ds.map(preprocess, batched=False)
# -

# ## Save

ds.to_parquet(data_dir / "preprocessed_224_crossvit_ds.pq")

# # Train models

# ## vit_base_patch16_224

model = CrossViTDualBranch("vit_base_patch16_224")

# +
ds = Dataset.from_parquet(str(data_dir / "preprocessed_224_crossvit_ds.pq"))
ds.set_format("torch", device="mps")
ds = ds.train_test_split(test_size=0.2)

train_dataloader = prepare_dataloader(ds, batch_size=1, split="train")
val_dataloader = prepare_dataloader(ds, batch_size=1, split="test")
# -

run_training(model, train_dataloader, val_dataloader, num_epochs=20, learning_rate=0.0001, device="mps")



max_value = max(df["label"])
max_value


def test_model(model, ds, max_value):
    train_df = ds["train"].to_pandas()
    print(f"Train set mean weight:\t{train_df['label'].mean() * max_value:.1f}kg")
    print(f"Train set weight range:\t{min(train_df['label']) * max_value}kg - {max(train_df['label']) * max_value}kg")
    print()

    model.eval()
    ds = ds["test"].map(lambda x: {"prediction": model(x["rgb"], x["depth"]).detach().cpu().numpy().flatten()}, batched=True, batch_size=8)
    df = ds.to_pandas()[["label", "prediction"]]

    df["prediction_in_kg"] = df["prediction"].map(lambda x: x * max_value)
    df["error"] = df.apply(lambda x: abs(x["label"] - x["prediction"]), axis=1)
    df["error_in_kg"] = df["error"].map(lambda x: x * max_value)
    df["%_error"] = df.apply(lambda x: x["error"] / x["label"] * 100, axis=1)

    print(f"Eval set weight range:\t{min(df['label']) * max_value}kg - {max(df['label']) * max_value}kg")
    print(f"Eval set mean weight:\t{df['label'].mean() * max_value:.1f}kg")
    print()
    print(f"Mean prediction in kg:\t{df['prediction'].mean() * max_value:.3f}")
    print(f"Mean error:\t\t{df['error'].mean():.3f}")
    print(f"Mean kg error:\t\t{df['error'].mean() * max_value:.1f}kg")
    print(f"Mean %error:\t\t{df['%_error'].mean():.1f}%")

    fig, ax = plt.subplots(3, 2, figsize=(16, 12))  # 3 rows, 2 columns

    # test set weight distribution
    sns.histplot(df["label"] * max_value, bins=100, kde=True, ax=ax[0,0])
    ax[0,0].set_title("Eval Set Weight Distribution", fontsize=16)
    ax[0,0].set_xlabel("Weight / kg", fontsize=12)
    ax[0,0].set_ylabel("Frequency", fontsize=12)

    # train set weight distribution
    sns.histplot(train_df["label"] * max_value, bins=100, kde=True, ax=ax[0,1])
    ax[0,1].set_title("Train Set Weight Distribution", fontsize=16)
    ax[0,1].set_xlabel("Weight / kg", fontsize=12)
    ax[0,1].set_ylabel("Frequency", fontsize=12)

    # predicted weight distribution
    sns.histplot(df["prediction_in_kg"], bins=100, kde=True, ax=ax[1,0])
    ax[1,0].set_title("Prediction Distribution", fontsize=16)
    ax[1,0].set_xlabel("Weight / kg", fontsize=12)
    ax[1,0].set_ylabel("Frequency", fontsize=12)

    # predicted vs actual
    ax[1,1].scatter(df["label"], df["prediction"], alpha=0.6)
    min_label = min(df["label"])
    max_label = max(df["label"])
    ax[1,1].plot([min_label, max_label], [min_label, max_label], 'r--')  # Diagonal line
    ax[1,1].set_title("Predicted vs. Actual Values")
    ax[1,1].set_xlabel("Labels")
    ax[1,1].set_ylabel("Predictions")

    # error distribution
    sns.histplot(df["error_in_kg"], bins=100, kde=True, ax=ax[2,0])
    ax[2,0].set_title("Error Distribution", fontsize=16)
    ax[2,0].set_xlabel("Error / kg", fontsize=12)
    ax[2,0].set_ylabel("Frequency", fontsize=12)

    # % error distribution
    sns.histplot(df["%_error"], bins=100, kde=True, ax=ax[2,1])
    ax[2,1].set_title("% Error Distribution", fontsize=16)
    ax[2,1].set_xlabel("Error / %", fontsize=12)
    ax[2,1].set_ylabel("Frequency", fontsize=12)

    plt.tight_layout()
    plt.show()
    return df


results_df = test_model(model, ds, max_value)

results_df[results_df["label"] > 600/max_value]

trimmed_df = results_df[results_df["label"] < 606/max_value]
print(f"Mean prediction in kg:\t{trimmed_df['prediction'].mean() * max_value:.3f}")
print(f"Mean error:\t\t{trimmed_df['error'].mean():.3f}")
print(f"Mean kg error:\t\t{trimmed_df['error'].mean() * max_value:.1f}kg")
print(f"Mean %error:\t\t{trimmed_df['%_error'].mean():.1f}%")



# # vit_large_patch16_224

model = CrossViTDualBranch("vit_large_patch16_224")

train_dataloader = prepare_dataloader(ds, batch_size=1, split="train")
val_dataloader = prepare_dataloader(ds, batch_size=1, split="test")

run_training(model, train_dataloader, val_dataloader, num_epochs=5, learning_rate=0.0001, device="mps")


