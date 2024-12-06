import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
from datasets import Dataset
from pathlib import Path
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from x_vit.crossvit import CrossViTDualBranch
from x_vit.utils import formatted_now


def prepare_dataloader(dataset, batch_size, num_workers=0, pin_memory=False, split="train"):
    def collate_fn(batch):
        rgb_images = torch.stack([item["rgb"] for item in batch])
        depth_images = torch.stack([item["depth"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
        return rgb_images, depth_images, labels

    return DataLoader(
        dataset[split], 
        batch_size=batch_size, 
        shuffle=(split == "train"), 
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers
    )

def run_training(
    model, 
    train_dataloader, 
    val_dataloader, 
    num_epochs=5, 
    learning_rate=1e-4, 
    device="mps",
    use_wand=True
):
    if use_wand:
        run = wandb.init(
            project="cows",
            config={
                "learning_rate": learning_rate,
                "architecture": "Dual branch CrossViT",
                "dataset": "CID",
                "epochs": num_epochs,
                "branch_base_model": model.branch_model_name,
            }
        )
        run.watch(model)
    
    # Move model to the specified device
    model = model.to(device)
    
    # # Set up optimizer and scheduler
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001,
        eps=1e-4,
        weight_decay=0.01
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    
    #optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=learning_rate, warmup_steps=4)

    for epoch in range(1, num_epochs+1):
        train_loss = train(model, train_dataloader, optimizer, epoch, num_epochs, device)
        val_loss = validate(model, val_dataloader, optimizer, epoch, num_epochs, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch}/{num_epochs}] - Training loss "
            f"{train_loss:.4f}. Validation loss {val_loss:.4f}. "
            f"Learning rate {lr:.3f}"
        )
        if use_wand:
            wandb.log({"learning_rate": lr}, step=epoch)
        
    run_output_dir = Path.home() / "trained_models" / formatted_now()
    run_output_dir.mkdir(exist_ok=True, parent=True)
    torch.save(model.state_dict(), run_output_dir / "model.pt")
    if use_wand:
        run.finish()


def validate(model, val_dataloader, optimizer, epoch, num_epochs, device):
    model.eval()
    if hasattr(optimizer, "eval"):
        optimizer.eval()
    val_loss = 0.0
    
    with torch.inference_mode():
        for images_1, images_2, target in tqdm(
            val_dataloader, 
            desc=f"Validation Epoch {epoch}/{num_epochs}", 
            ncols=120
        ):
            images_1 = images_1.to(device)
            images_2 = images_2.to(device)
            target = target.to(device)
            
            outputs = model(images_1, images_2).squeeze()
            loss = F.mse_loss(outputs, target)
            val_loss += loss.item()
            
            # Free MPS memory, but slightly slower
            if device == "mps":
                images_1 = images_1.cpu()
                images_2 = images_2.cpu()
                target = target.cpu()
                outputs = outputs.cpu()
                loss = loss.cpu()
    
    return val_loss / len(val_dataloader)


def train(model, train_loader, optimizer, epoch, num_epochs, device):
    model.train()
    if hasattr(optimizer, "train"):
        optimizer.train()
    train_loss = 0.0
    
    for images_1, images_2, target in tqdm(
        train_loader, desc=f"Training Epoch {epoch}/{num_epochs}", ncols=120
    ):
        # * Create MPS computation block
        with torch.device(device):
            images_1 = images_1.to(device)
            images_2 = images_2.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(images_1, images_2).squeeze()
            loss = F.mse_loss(outputs, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # * clear intermediate tensors
            if device == "mps":
                del images_1, images_2, outputs
                torch.mps.empty_cache()

    return train_loss / len(train_loader)

if __name__ == "__main__":
    USE_WAND = False
    data_dir = Path.home() / "repos/weigh-protein/data"
    ds = Dataset.from_parquet(str(data_dir / "preprocessed_224_crossvit_ds.pq"))
    ds.set_format("torch", device="mps")
    ds = ds.train_test_split(test_size=0.2)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # * Enable torch.compile with MPS focused settings
    model = CrossViTDualBranch("vit_base_patch16_224")
    model = torch.compile(
        model,
        backend="eager",
        # mode="reduce-overhead"
    )
    
    batch_size = 12
    NUM_WORKERS = 0
    PIN_MEMORY= True # * Faster transfer to GPU

    train_dataloader = prepare_dataloader(
        ds, 
        batch_size=batch_size, 
        split="train",
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY  
    )
    val_dataloader = prepare_dataloader(
        ds, 
        batch_size=batch_size, 
        split="test",
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    
    run_training(
        model, 
        train_dataloader, 
        val_dataloader, 
        num_epochs=2, 
        learning_rate=0.0001, 
        device=device,
        use_wand=USE_WAND
    )