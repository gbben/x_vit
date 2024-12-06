import json
import os
from datetime import datetime
from pathlib import Path

import torch
import shutil
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import create_repo, HfApi
from PIL import Image
from safetensors.torch import load_file as safetensors_load_file
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTModel, TrainingArguments, Trainer


class ViTRegressionModel(nn.Module):
    def __init__(self, base_model: str):
        super(ViTRegressionModel, self).__init__()
        self.vit = ViTModel.from_pretrained(
            base_model,
            ignore_mismatched_sizes=True,
            add_pooling_layer=False,
        )
        self.regression_head = nn.Linear(self.vit.config.hidden_size, 1)

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token
        values = self.regression_head(cls_output)
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(values.view(-1), labels.view(-1))
        return (loss, values) if loss is not None else values


def train_model(
    base_model: str,
    image_size: int,
    dataset_path: str | Path,
    value_column_name: str,
    test_split: int,
    output_dir: str | Path,
    num_train_epochs: int,
    learning_rate: float,
) -> Path:
    # Load the dataset
    dataset_path = str(dataset_path)
    dataset = Dataset.from_parquet(dataset_path)

    # Split the dataset into train and test
    dataset = dataset.train_test_split(test_size=test_split)

    # Create output dir with timestamp
    now = datetime.now().strftime("%Y-%m-%d__%H-%M")
    output_dir = Path(output_dir) / now

    # Save train and test datasets
    dataset["train"].to_parquet(output_dir / "train_dataset.pq")
    dataset["test"].to_parquet(output_dir / "test_dataset.pq")

    # Get max value
    train_values = dataset['train'][value_column_name]
    test_values = dataset['test'][value_column_name]
    max_value = max(train_values + test_values)
    print('Max Value:', max_value)

    # Define a transform to convert PIL images to tensors
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    def preprocess(example):
        example['image'] = transform(example['image'])
        example[value_column_name] = example[value_column_name] / max_value  # Normalize values
        return example

    # Apply the preprocessing with normalization
    dataset = dataset.map(preprocess, batched=False)
    dataset.set_format("torch", device="mps")

    def collate_fn(batch):
        # Ensure that each item['image'] is a tensor
        pixel_values = torch.stack([x["image"] for x in batch])  # [torch.tensor(item['image']) for item in batch])
        pixel_values.to("mps")
        labels = torch.tensor([item[value_column_name] for item in batch], dtype=torch.float).unsqueeze(1)
        labels.to("mps")
        return {'pixel_values': pixel_values, 'labels': labels}

    model = ViTRegressionModel(base_model)
    model.to("mps")

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        save_steps=10,
        save_total_limit=1,
        logging_steps=10,
        remove_unused_columns=False,
        resume_from_checkpoint=True,
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=collate_fn,
    )

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        labels = p.label_ids
        mse = ((preds - labels) ** 2).mean().item()
        return {"mse": mse}

    trainer.compute_metrics = compute_metrics

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Write jSON file
    data = {
        "base_model": base_model,
        "dataset_path": dataset_path,
        "value_column_name": value_column_name,
        "test_split": test_split,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "max_value": max_value,
    }
    filename = 'metadata.json'
    # Traverse the directory tree starting from the current directory
    for root, dirs, files in os.walk(output_dir):
        for dir_name in dirs:
            if 'checkpoint' in dir_name:
                # Construct the full path to the target directory
                dir_path = os.path.join(root, dir_name)
                # Construct the full path to the JSON file in the target directory
                file_path = os.path.join(dir_path, filename)
                # Write the JSON data to the file
                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=4)
                print(f'Data successfully written to {file_path}')

    return output_dir


def predict(image: Image, image_size: int, model_dir: str):
    # get max value and base model name
    with open(f"{model_dir}/metadata.json") as f:
        data = json.load(f)
    max_value = data["max_value"]
    base_model = data.get("base_model", "google/vit-base-patch16-224")  # initial runs did not save this
    
    # Load the saved model checkpoint
    model = ViTRegressionModel(base_model)
    model.to("mps")
    checkpoint_path = f"{model_dir}/model.safetensors"
    state_dict = safetensors_load_file(checkpoint_path)

    # original runs kept these unused weights
    if "vit.pooler.dense.bias" in state_dict:
        state_dict.pop("vit.pooler.dense.bias")
        state_dict.pop("vit.pooler.dense.weight")

    # original runs had bad name for regression head
    if "classifier.bias" in state_dict:
        state_dict["regression_head.bias"] = state_dict.pop("classifier.bias")
        state_dict["regression_head.weight"] = state_dict.pop("classifier.weight")
    
    model.load_state_dict(state_dict)
    model.eval()

    # Define a transform to convert PIL images to tensors
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Preprocess the image
    image = transform(image).unsqueeze(0).to("mps")  # Add batch dimension

    with torch.no_grad():
        # Run the model
        prediction = model(image)

    # De-normalize the prediction
    prediction = prediction.item() * max_value
    return prediction
