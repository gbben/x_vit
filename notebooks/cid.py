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

# # Experimenting with FastSAM and DepthAnythingV2

# %load_ext autoreload
# %autoreload 2

# ## Load CID

from pathlib import Path
import pandas as pd
from constants import *

cid_df = pd.read_csv(cid_dataset_csv)
cid_df

images = list(cid_images_dir.glob("*/*.jpg"))
print(f"{len(images)} images")

yt_images = list(cid_yt_images_dir.glob("*/*.jpg"))
print(f"{len(yt_images)} yt images")

# <br>

# ## FastSAM

# ### Example

import math
import numpy as np
from PIL import Image
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt

# +
source = images[0]

with Image.open(source) as image:
    print(image.size)
    max_dim = max(image.size)
    imgsz = 32 * math.ceil(max_dim / 32)

# +
fastsam = FastSAM(sam_model_file)
everything_results = fastsam(source, device="mps", retina_masks=True, imgsz=imgsz, conf=0.3, iou=0.9)

prompt_process = FastSAMPrompt(source, everything_results, device="mps")

# +
# everything prompt
ann = prompt_process.everything_prompt()
prompt_process.plot(annotations=ann, output=output_dir)

with Image.open(output_dir / source.name) as img:
    display(img)

# +
# text prompt
ann = prompt_process.text_prompt(text="a cow")
prompt_process.plot(annotations=ann, output=output_dir)

with Image.open(output_dir / source.name) as img:
    display(img)
# -

cow_mask = ann[0].masks.data.detach().cpu().numpy().transpose(1, 2, 0)
original_image = ann[0].orig_img

masked = np.where(cow_mask, original_image, 0)

Image.fromarray(np.array(masked).astype(np.uint8))

# **Note**: no fucking clue why this cow went blue, but I thoroughly enjoyed when it rendered.

# <br>

# ### Functions

import torch


def generate_cow_mask(relative_image_path: Path | str) -> torch.Tensor:
    image_path = cid_dir / relative_image_path
    
    with Image.open(image_path) as image:
        max_dim = max(image.size)
    imgsz = 32 * math.ceil(max_dim / 32)
    
    everything_results = fastsam(
        image_path,
        device="mps",
        retina_masks=True,
        imgsz=imgsz,
        conf=0.3,
        iou=0.9,
        verbose=False,
    )
    
    prompt_process = FastSAMPrompt(image_path, everything_results, device="mps")
    ann = prompt_process.text_prompt(text="a brown cow")
    masks = ann[0].masks.data.detach().cpu().long().numpy()

    if len(masks) != 1:
        print(f"unexpected number of masks: [len(masks)]")

    return masks


def batch_generate_cow_mask(relative_image_paths: list[Path | str]) -> dict:
    image_paths = [cid_dir / x for x in relative_image_paths]

    imgsz = 0
    for image_path in image_paths:
        with Image.open(image_path) as image:
            max_dim = max(image.size)
        imgsz = max(32 * math.ceil(max_dim / 32), imgsz)

    everything_results = fastsam(
        image_paths,
        device="mps",
        retina_masks=True,
        imgsz=imgsz,
        conf=0.3,
        iou=0.9,
        verbose=False
    )
    
    prompt_process = FastSAMPrompt(image_path, everything_results, device="mps")
    ann = prompt_process.text_prompt(text="a brown cow")
    masks = [x.masks.data.detach().cpu().long().numpy() for x in ann]

    return {"cow_mask": masks}


# <br>

# ## Depth Anything V2

# ### Example

# #### Base model

from transformers import pipeline

depth_pipeline = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="mps")

image = Image.open(source)

result = depth_pipeline(image)
result["depth"]

result["predicted_depth"]

result["predicted_depth"].shape

# **Need to ensure the size is the same as the input size.**

# <br>

# #### Metric depth model
# For this we have to use the [DepthAnythingV2 repo](https://github.com/DepthAnything/Depth-Anything-V2) itself, as `transformers` does not support the metric depth model variant.

import cv2
import torch
from weigh_protein.libs.depth_anything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

da2_metric_depth_model_file

da2_model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}
}
da2_encoder = "vitb"  # 'vitl' or 'vits' or 'vitb'
da2_max_depth = 20  # 20 for indoor model, 80 for outdoor model

da2_model = DepthAnythingV2(max_depth=da2_max_depth, **da2_model_configs[da2_encoder])

da2_model.load_state_dict(torch.load(da2_metric_depth_model_file, map_location="mps"))
da2_model.eval()
da2_model = da2_model.to("mps")  # suppress output

raw_img = cv2.imread(source)
depth_mask = da2_model.infer_image(raw_img)  # HxW depth map in meters in numpy


# <br>

# ### Functions

# #### Base model

def generate_depth_mask(relative_image_path: Path | str):
    image_path = cid_dir / relative_image_path
    
    depth_pipeline = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Base-hf",
        device="mps",
    )

    with Image.open(image_path) as image:
        result = depth_pipeline(image)

    return result["predicted_depth"]


# #### Metric depth model

def generate_metric_depth_mask(relative_image_path: Path | str) -> np.array:
    """Generate a metric depth mask using DepthAnythingV2.

    Args:
        relative_image_path (Path|str): path to the source image,
            relative to the cid directory.

    Returns:
        np.array: metric depth mask with width and height matching the
            source image.
    """
    image_path = cid_dir / relative_image_path
    raw_img = cv2.imread(image_path)
    return da2_model.infer_image(raw_img)


# <br>

# <br>

# ## YOLO-World + SAM2

# ### Example

# #### YOLO 

# +
# Import YOLOWorld class from ultralytics module
from ultralytics import YOLOWorld

# Initialize the model with pre-trained weights
yolo_model = YOLOWorld("../models/yolov8s-world.pt", verbose=False)

# Set the classes you'd like to find in your image
yolo_model.set_classes(["cow"])
# -

# %%time
results = yolo_model.predict(source, max_det=100, iou=0.01, conf=0.01, verbose=False)

box = results[0].boxes[0].xyxy.numpy()[0]
box

# #### SAM2

import matplotlib.pyplot as plt
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# +
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


# -

sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-base-plus", device="mps")

# TODO try coreml version
sam2_coreml_predictor = "apple/coreml-sam2.1-large"

# %%time
with torch.inference_mode(), Image.open(source) as image:
    sam2_predictor.set_image(image)
    sam2_masks, sam2_scores, _ = predictor.predict(box=box, multimask_output=False)

show_masks(image, sam2_masks, sam2_scores, box_coords=box)

# <br>

# ### Functions

type(sam2_masks[0])


def generate_cow_mask_sam2(image_path: str):
    yolo_results = yolo_model.predict(image_path, max_det=100, iou=0.01, conf=0.01, verbose=False)
    cow_bbox = yolo_results[0].boxes[0].xyxy.numpy()[0]
    with torch.inference_mode(), Image.open(image_path) as image:
        sam2_predictor.set_image(image)
        sam2_masks, _, _ = sam2_predictor.predict(box=cow_bbox, multimask_output=False)
    return sam2_masks[0]


# <br>

# ## Combine depth and cow masks

# ### FastSAM example

cow_mask.shape

depth_mask.shape

cow_masked_depth = np.where(cow_mask[:, :, 0], depth_mask, 0)
cow_masked_depth

cmap = "viridis"

plt.figure(figsize=(15,10))
plt.imshow(cow_masked_depth, cmap=cmap, interpolation="none")
plt.colorbar(shrink=0.8)
plt.show()


# This gives us pixel-by-pixel metric depth map for the detected cow. We can use this to extract some features.

# ### Functions

def combine_cow_and_depth_masks(cow_mask: np.array, depth_mask: np.array) -> np.array:
    """Combine the cow mask with the depth mask.

    Args:
        cow_mask (np.array): output array from FastSAM cast to a binary
            numpy array.
        depth_mask (np.array): h x w numpy array of metric depth.

    Returns:
        np.array: Cow-shaped depth mask.
    """
    cow_mask = cow_mask.transpose(1, 2, 0)
    return np.where(cow_mask[:, :, 0], depth_mask, 0)


def combine_sam2_and_depth_masks(sam2_mask: np.array, depth_mask: np.array) -> np.array:
    """Combine the sam2 cow mask with the depth mask.

    Args:
        sam2_cow_mask (np.array): h x w numpy array from SAM2.
        depth_mask (np.array): h x w numpy array of metric depth.

    Returns:
        np.array: Cow-shaped depth mask.
    """
    return np.where(sam2_mask, depth_mask, 0)


# ### SAM2 Example

sam2_metric_combo = combine_sam2_and_depth_masks(sam2_masks[0], depth_mask)

plt.figure(figsize=(15,10))
plt.imshow(sam2_metric_combo, cmap=cmap, interpolation="none")
plt.colorbar(shrink=0.8)
plt.show()


# ## Pipelines

# ### SAM2 + DAv2

def generate_sam2_da2_combined_mask(image_path: str):
    sam2_mask = generate_cow_mask_sam2(image_path)
    da_mask = generate_metric_depth_mask(image_path)
    return combine_sam2_and_depth_masks(sam2_mask, depth_mask)


# #### Test

# %time
result = generate_sam2_da2_combined_mask(source)

plt.figure(figsize=(15,10))
plt.imshow(result, cmap=cmap, interpolation="none")
plt.colorbar(shrink=0.8)
plt.show()

# <br>

# ## Feature extraction

# ### Explore

non_zero_cow_masked_depth = np.nonzero(cow_masked_depth)
nearest_point = np.min(cow_masked_depth[non_zero_cow_masked_depth])
nearest_point

np.where(cow_masked_depth == nearest_point)

furthest_point = np.max(cow_masked_depth)
furthest_point

np.where(cow_masked_depth == furthest_point)

uppermost_pixel_y = np.min(non_zero_cow_masked_depth[0])
uppermost_pixel_y

bottommost_pixel_y = np.max(non_zero_cow_masked_depth[0])
bottommost_pixel_y

leftmost_pixel_x = np.min(non_zero_cow_masked_depth[1])
leftmost_pixel_x

rightmost_pixel_x = np.max(non_zero_cow_masked_depth[1])
rightmost_pixel_x

plt.figure(figsize=(15,10))
plt.imshow(cow_masked_depth, cmap=cmap, interpolation="none")
plt.colorbar(shrink=0.6)
plt.axvline(x=leftmost_pixel_x, color="red")
plt.axvline(x=rightmost_pixel_x, color="red")
plt.axhline(y=uppermost_pixel_y, color="red")
plt.axhline(y=bottommost_pixel_y, color="red")
plt.show()

# We may need to consider some techniques to ensure we aren't getting depths of non-cow pixels. The above calculations use the absolute max and mins along each axis, which doesn't leave much room for error in the segmentation performance. We can try to obtain more reliable values by taking averages over different proportions of the cow, such as the mean of the masked depth values for the front and back fifths of the cow.

width_of_a_fifth = (rightmost_pixel_x - leftmost_pixel_x) * 0.2
left_fifth_x1 = int(leftmost_pixel_x + width_of_a_fifth)
left_fifth_x1

right_fifth_x0 = int(rightmost_pixel_x - width_of_a_fifth)
right_fifth_x0

plt.figure(figsize=(15,10))
plt.imshow(cow_masked_depth, cmap=cmap, interpolation="none")
plt.colorbar(shrink=0.6)
plt.axvspan(leftmost_pixel_x, left_fifth_x1, color="red", alpha=0.3)
plt.axvspan(right_fifth_x0, rightmost_pixel_x, color="red", alpha=0.3) 
plt.show()

left_fifth = cow_masked_depth[:, leftmost_pixel_x:left_fifth_x1 + 1]
non_zero_left_fifth = left_fifth[left_fifth != 0]
mean_left_fifth = np.mean(non_zero_left_fifth)
mean_left_fifth

right_fifth = cow_masked_depth[:, right_fifth_x0:rightmost_pixel_x + 1]
non_zero_right_fifth = right_fifth[right_fifth != 0]
mean_right_fifth = np.mean(non_zero_right_fifth)
mean_right_fifth


# <br>

# ### Functions

def calculate_mean_depth_of_left_and_right_segments(mask: np.array, n_segments: int = 5) -> tuple[float, float]:
    non_zero_mask = np.nonzero(mask)

    # calculate width of a segment
    leftmost_pixel_x = np.min(non_zero_mask[1])
    rightmost_pixel_x = np.max(non_zero_mask[1])
    width_of_segment = int((rightmost_pixel_x - leftmost_pixel_x) / n_segments)

    # calculate width of left segment
    left_segment_x1 = leftmost_pixel_x + width_of_segment
    left_segment = mask[:, leftmost_pixel_x:left_segment_x1 + 1]
    non_zero_left_segment = left_segment[left_segment != 0]
    mean_left_segment = np.mean(non_zero_left_segment)

    # calculate width of right segment
    right_segment_x0 = rightmost_pixel_x - width_of_segment
    right_segment = mask[:, right_segment_x0:rightmost_pixel_x + 1]
    non_zero_right_segment = right_segment[right_segment != 0]
    mean_right_segment = np.mean(non_zero_right_segment)

    return mean_left_segment, mean_right_segment


def calculate_mean_depth(mask: np.array) -> float:
    non_zero = mask[mask != 0]
    return np.mean(non_zero)


def calculate_height_in_pixels(mask) -> int:
    non_zero_mask = np.nonzero(mask)
    uppermost_pixel_y = np.min(non_zero_mask[0])
    bottommost_pixel_y = np.max(non_zero_mask[0])
    return bottommost_pixel_y - uppermost_pixel_y


def calculate_length_in_pixels(mask) -> int:
    non_zero_mask = np.nonzero(mask)
    leftmost_pixel_x = np.min(non_zero_mask[1])
    rightmost_pixel_x = np.max(non_zero_mask[1])
    return rightmost_pixel_x - leftmost_pixel_x


# <br>

# ## Plan

# ### Baseline
# - create a pipeline using only the features in the CID csv
# - cross validate models
# - this will provide a baseline to beat with our additional features

# ### First iteration
# - create a dataset from the `images` directory (2056 images for 514 cows)
# - start by only using the side angle with the cow facing left (514 images, one quarter of the images in the `images` directory)
# - run semantic segmentation for each image
# - run metric depth estimation for each image
# - calculate mean of all non-zero depths for cow distance. cow is not at an angle so no need to do front and back segments
# - count number of cow pixels and calculate fraction of total pixels
# - calculate cow length using cow height, length in pixels and height in pixels
# - cross validate models

# ### Second iteration

# - expand on first iteration by adding in one of the angled images for each of the cows
# - calculate back and front segment mean distances and number of pixels for the angled images
# - cross validate models using features from both images

# <br>

# ## Evaluation functions

from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


def cross_validate_pipeline(*, X, y, pipeline, cv, print_results=False) -> dict:
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_squared_error")
    rmse_scores = (-scores) ** 0.5
    rmse = round(rmse_scores.mean(), 2)
    rmse_std = round(rmse_scores.std(), 2)
    mean_y = round(np.mean(y), 2)
    nrmse_percentage = round(rmse / mean_y * 100, 2)
    nrmse_percentage_std = round(rmse_std / mean_y * 100, 2)
    
    if print_results:
        print(f"RMSE: {rmse} ± {rmse_std}")
        print(f"Mean of actual values: {mean_y}")
        print(f"NRMSE: {nrmse_percentage} ± {nrmse_percentage_std}%")

    return {
        "rmse/kg": rmse,
        "rmse_std/kg": rmse_std,
        "nrmse/%": nrmse_percentage,
        "nrmse_std/%": nrmse_percentage_std,
    }


# <br>

# ## Baseline

# ### Define target variable

target = "weight_in_kg"

cid_df[target]

# <br>

# ### Preprocessing pipeline

# #### Select features

cid_df.columns

baseline_features = [
    "sex",
    "color",
    "breed",
    "age_in_year",
    "height_in_inch",
    "size",
]

cid_df[baseline_features]


# #### Define preprocessor

def create_baseline_preprocessor() -> ColumnTransformer:
    numeric_features = ["age_in_year", "height_in_inch"]
    categorical_features = ["sex", "color", "breed", "size"]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(drop="first"))])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


# <br>

# ### Evaluate models

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "Neural Network": MLPRegressor(max_iter=10000, random_state=42),
}

baseline_preprocessor = create_baseline_preprocessor()

baseline_scores = pd.DataFrame.from_dict(
    {
        model_name: cross_validate_pipeline(
            X=cid_df[baseline_features],
            y=cid_df[target],
            pipeline=Pipeline(
                steps=[
                    ("preprocessor", baseline_preprocessor),
                    ("regressor", model)
                ]
            ),
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
        )
        for model_name, model in models.items()
    },
    orient="index",
)
baseline_scores

# <br>

# ## First iteration

# ### Create image dataframe

image_df = pd.DataFrame({"path": sorted(x.relative_to(cid_dir) for x in images)})
image_df

# add sku column to be able to join with cid df and convert path column to string
image_df["sku"] = image_df["path"].map(lambda x: x.parts[1])
image_df["path"] = image_df["path"].map(str)
image_df

# <br>

# ### Create NN regression Dataset

from datasets import Dataset, Image as HFImage
from weigh_protein.libs.image_regression.image_regression import predict

pq_location = output_dir / "image_regression/test_dataset.pq"
ds = Dataset.from_parquet(str(pq_location))
model_dir =  output_dir / "image_regression/checkpoint-232"

ds = ds.map(lambda x: {"prediction2": predict(x["image"], model_dir)})

prediction_results_df = ds.remove_columns("image").to_pandas()

prediction_results_df["diff"] = prediction_results_df.apply(lambda x: x["prediction"] - x["weight_in_kg"], axis=1).map(abs)
prediction_results_df["diff"].mean()

prediction_results_df["diff2"] = prediction_results_df.apply(lambda x: x["prediction2"] - x["weight_in_kg"], axis=1).map(abs)
prediction_results_df["diff2"].mean()

prediction_results_df["weight_in_kg"].mean()



# #### Longer run, 5 epochs

pq_location = output_dir / "image_regression/24-10-20_19-45/test_dataset.pq"
ds = Dataset.from_parquet(str(pq_location))
model_dir =  output_dir / "image_regression/24-10-20_19-45/checkpoint-1160"

ds = ds.map(lambda x: {"prediction": predict(x["image"], model_dir)})

prediction_results_df = ds.remove_columns("image").to_pandas()

prediction_results_df["diff"] = prediction_results_df.apply(lambda x: x["prediction"] - x["weight_in_kg"], axis=1).map(abs)
prediction_results_df["diff"].mean()

prediction_results_df["%_diff"] = prediction_results_df.apply(lambda x: x["diff"] / x["weight_in_kg"] * 100, axis=1)
prediction_results_df["%_diff"].mean()

# +
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the distribution using seaborn's distplot
plt.figure(figsize=(8, 6))

# Seaborn's kde and histogram plot
sns.histplot(prediction_results_df["diff"], bins=100, kde=True)

# Adding labels and title
plt.title("Distribution of 'diff'", fontsize=16)
plt.xlabel("Values", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

# Display the plot
plt.show()
# -



regression_df = pd.merge(image_df, cid_df[["sku", "weight_in_kg"]], on="sku", how="inner").drop(columns="sku")
regression_df["image"] = regression_df["path"].map(
    lambda x: cid_dir / x
).map(
    str
).map(
    Image.open
)
regression_df["file_name"] = regression_df["path"].map(lambda x: x.lstrip("images/"))
regression_df

regression_df[["file_name", "weight_in_kg"]].to_json("../data/cid/images/metadata.jsonl", lines=True, orient="records")

temp_dict = regression_df[["image", "weight_in_kg"]].to_dict(orient="list")
dataset = Dataset.from_dict(temp_dict)
dataset[0]

dataset.to_parquet(cid_dir / "hf_regression_ds.pq")

# <br>

# ### Create left side image dataframe

left_side_df = image_df[image_df["path"].str.endswith("0.jpg")].reset_index().drop(columns="index")
left_side_df

# <br>

# ### Generate segmentation masks
# Now we will use FastSAM to detect the cow in each image and save the masks.

# #### Serial

from datasets import Dataset

# %%time
left_side_df[:6]["path"].map(generate_cow_mask)

# %%time
Dataset.from_pandas(left_side_df).select(range(6)).map(
    lambda x: {"cow_mask": generate_cow_mask(x)}, input_columns="path", batched=False
)

left_side_df["cow_mask"] = left_side_df["path"].map(generate_cow_mask)

# #### Batch [CAUSED ISSUES WHEN SAVING - ABANDONED FOR NOW]
# We can use a huggingface `Dataset` to perform batch processing

# +
#left_side_ds = Dataset.from_pandas(left_side_df)
# -

# ##### Test run:

# +
# #%%time
#left_side_ds.select(range(6)).map(batch_generate_cow_mask, input_columns="path", batched=True, num_proc=1, batch_size=3)
# -

# **Note**: Using more than 1 process causes a fatal crash.

# ##### Run everything

# +
# #%%time
#left_side_ds = left_side_ds.map(batch_generate_cow_mask, input_columns="path", batched=True, num_proc=1, batch_size=3)

# +
#left_side_ds.to_parquet("cow_mask.pq")
# -

# <br>

# ### Save cow masks

temp = left_side_df.copy()
temp["cow_mask"] = temp["cow_mask"].map(lambda x: x.tolist())
temp.to_parquet(cid_left_side_cow_masks_pq)

# <br>

# ### Generate depth masks

# Batch processing would require changes to the `DepthAnythingV2` class - not worth the time at this stage.

# +
#left_side_ds.select(range(4)).map(lambda x: {"depth_mask": generate_metric_depth_mask(x)}, input_columns="path", batched=False)

# +
# fails with error "ArrowInvalid: offset overflow while concatenating arrays"
# not investigating for now
#left_side_ds = left_side_ds.map(lambda x: {"depth_mask": generate_metric_depth_mask(x)}, input_columns="path", batched=False)
# -

# We shall use the DataFrame due to issues with using Dataset's .map method and because batch processing is not supported.

# %%time
left_side_df["depth_mask"] = left_side_df["path"].map(generate_metric_depth_mask)

left_side_df


# <br>

# ### Fix arrays [NO LONGER REQUIRED]

# Array got screwed up by dataset. Fix required.

def fix_dataset_noncery(cow_mask: np.array) -> np.array:
    return np.array([np.stack(cow_mask[0])])


# +
#left_side_df["cow_mask"] = left_side_df["cow_mask"].map(fix_dataset_noncery)
# -

# <br>

# ### Combine masks

example = left_side_df.loc[0]
combine_cow_and_depth_masks(example["cow_mask"], example["depth_mask"])

left_side_df["combined_mask"] = left_side_df.apply(lambda x: combine_cow_and_depth_masks(x["cow_mask"], x["depth_mask"]), axis=1)
left_side_df

# <br>

# ### Plot a few samples

for mask in left_side_df.sample(4)["combined_mask"]:
    plt.figure(figsize=(15,10))
    plt.imshow(mask, cmap=cmap, interpolation="none")
    plt.colorbar(shrink=0.8)
    plt.show()

# <br>

# ### Calculate mean depth

left_side_df["mean_depth"] = left_side_df["combined_mask"].map(calculate_mean_depth)
left_side_df

# <br>

# ### AUC (Area Under Cow) in pixels

left_side_df["cow_pixels"] = left_side_df["cow_mask"].map(np.sum)
left_side_df["cow_pixels_fraction"] = left_side_df.apply(lambda x: x["cow_pixels"] / x["cow_mask"].size, axis=1)
left_side_df


# wtf mate
def normalize_cow_pixels(row: dict) -> int:
    return int(row["cow_pixels"] * row["mean_depth"]**2)


# wtf mate
def normalize_cow_pixels_fraction(row: dict) -> float:
    return row["cow_pixels_fraction"] * row["mean_depth"]**2


left_side_df["normalized_cow_pixels"] = left_side_df.apply(normalize_cow_pixels, axis=1)
left_side_df["normalized_pixels_fraction"] = left_side_df.apply(normalize_cow_pixels_fraction, axis=1)
left_side_df

# <br>

# ### Cow length

left_side_df["length_in_pixels"] = left_side_df["combined_mask"].map(calculate_length_in_pixels)
left_side_df["height_in_pixels"] = left_side_df["combined_mask"].map(calculate_height_in_pixels)
left_side_df

# ### Join with CID dataframe

cid_df.head(1)

set(left_side_df["sku"].unique()) - set(cid_df["sku"].unique())

left_side_df = left_side_df.merge(cid_df, on="sku", how="inner")
left_side_df


# <br>

# ### Calculate length in inches

# We have the cow height in inches, and we have both the length and height in pixels. We will now calculate the length in inches using the other values.

def calculate_length_in_inches(row: dict) -> float:
    inches_per_pixel = row["height_in_inch"] / row["height_in_pixels"]
    return round(row["length_in_pixels"] * inches_per_pixel, 1)


left_side_df["length_in_inch"] = left_side_df.apply(calculate_length_in_inches, axis=1)


# <br>

# ### AUC in square inches

def calculate_auc_in_square_inches(row: dict) -> float:
    inches_per_pixel = row["height_in_inch"] / row["height_in_pixels"]
    square_inches_per_pixel = inches_per_pixel**2
    return round(row["cow_pixels"] * square_inches_per_pixel, 1)


left_side_df["area_under_cow_in_sq_inch"] = left_side_df.apply(calculate_auc_in_square_inches, axis=1)

# <br>

# ### Save dataset
# We drop the masks as these are large and we have already derived features from them.

left_side_df.drop(columns=["cow_mask", "depth_mask", "combined_mask"]).to_parquet(cid_left_side_dataset_pq)

left_side_df = pd.read_parquet(cid_left_side_dataset_pq)

# <br>

# ### Preprocessing pipeline

# #### Select features
# For now, we will select a subset of features for the first experiments. Later we can be more exhaustive.

left_side_df.columns

first_iteration_features = [
    "normalized_pixels_fraction",
    "sex",
    "color",
    "breed",
    "age_in_year",
    "height_in_inch",
    "length_in_inch",
    "size",
    "area_under_cow_in_square_inches",
]

left_side_df[first_iteration_features]


# #### Define preprocessor

def create_first_iteration_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "normalized_pixels_fraction",
        "age_in_year",
        "height_in_inch",
        "length_in_inch",
        "area_under_cow_in_square_inches",
    ]
    categorical_features = ["sex", "color", "breed", "size"]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(drop="first"))])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


# <br>

# ### Evaluate models

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "Neural Network": MLPRegressor(max_iter=10000, random_state=42),
}

first_iteration_preprocessor = create_first_iteration_preprocessor()

first_iteration_scores = pd.DataFrame.from_dict(
    {
        model_name: cross_validate_pipeline(
            X=df[first_iteration_features],
            y=df[target],
            pipeline=Pipeline(
                steps=[
                    ("preprocessor", first_iteration_preprocessor),
                    ("regressor", model)
                ]
            ),
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
        )
        for model_name, model in models.items()
    },
    orient="index",
)
first_iteration_scores

# <br>

# ## Second iteration
# Now we will add in another image for each cow, taken from a second angle.

# ### Create dataframe with left and front left images

image_df

two_sides_df = pd.concat(
    [
        image_df[image_df["path"].str.endswith("0.jpg")],
        image_df[image_df["path"].str.endswith("1.jpg")],
    ]
).sort_index()
two_sides_df["abs_path"] = two_sides_df["path"].map(lambda x: cid_dir / x).map(str)
two_sides_df

# <br>

# ### Generate segmentation masks
# This time we will use SAM2 to detect the cow in each image and save the masks.

from time import time

start = time()
two_sides_df["abs_path"].loc[:2].map(generate_sam2_da2_combined_mask)
print(f"Runtime: {time() - start:.1f}s")

start = time()
two_sides_df["cow_mask"] = two_sides_df["abs_path"].map(generate_cow_mask_sam2)
print(f"Runtime: {time() - start:.1f}s")

# ### Generate depth masks

start = time()
two_sides_df["depth_mask"] = two_sides_df["abs_path"].map(generate_metric_depth_mask)
print(f"Runtime: {time() - start:.1f}s")

# ### Generate combined masks

start = time()
two_sides_df["combined_mask"] = two_sides_df.apply(
    lambda x: combine_sam2_and_depth_masks(x["cow_mask"], x["depth_mask"]), axis=1
)
print(f"Runtime: {time() - start:.1f}s")

# ### More features

# AUC
two_sides_df["cow_pixels"] = two_sides_df["cow_mask"].map(np.sum)
two_sides_df["cow_pixels_fraction"] = two_sides_df.apply(lambda x: x["cow_pixels"] / x["cow_mask"].size, axis=1)

two_sides_df["mean_depth"] = two_sides_df["combined_mask"].map(calculate_mean_depth)

two_sides_df["length_in_pixels"] = two_sides_df["combined_mask"].map(calculate_length_in_pixels)
two_sides_df["height_in_pixels"] = two_sides_df["combined_mask"].map(calculate_height_in_pixels)
two_sides_df

two_sides_df = two_sides_df.merge(cid_df, on="sku", how="inner")
two_sides_df

two_sides_df["length_in_inch"] = two_sides_df.apply(calculate_length_in_inches, axis=1)

two_sides_df["area_under_cow_in_sq_inch"] = two_sides_df.apply(calculate_auc_in_square_inches, axis=1)

# ### Save full dataframe as pickle

cid_two_sides_df_pkl = cid_dir / "two_sides_df.pkl"

two_sides_df.to_pickle(cid_two_sides_df_pkl)

# ### Save features

two_sides_df.drop(columns=["abs_path", "cow_mask", "depth_mask", "combined_mask"]).to_parquet(cid_two_sides_dataset_pq)

two_sides_features_df = pd.read_parquet(cid_two_sides_dataset_pq)

# ### Combine two sides into one row

two_sides_features_df = two_sides_features_df.drop(columns=["path", "price", "images_count", "yt_images_count", "total_images"])

two_sides_features_df.head()

# +
grouped = two_sides_features_df.groupby("sku")
merged = []

# Iterate over the groups (one group per cow)
for name, group in grouped:
    # Create a dictionary to hold the merged data for this cow
    merged_data = {"sku": name}
    
    # Iterate through columns (skip the cow_name column)
    for col in group.columns:
        if col == "sku":
            continue
        
        # If all values in the group are the same for this column, keep a single column
        if group[col].nunique() == 1:
            merged_data[col] = group[col].iloc[0]
        else:
            # If values differ, keep both columns with unique names
            for idx, value in enumerate(group[col]):
                merged_data[f"{col}_side_{idx}"] = value
    
    # Append the merged row to the final DataFrame
    merged.append(merged_data)

combined_sides_features_df = pd.DataFrame(merged)
combined_sides_features_df
# -

# ### Evaluate models

# #### Select features

combined_sides_features_df.columns

second_iteration_features = [
    "sex",
    "color",
    "breed",
    "size",
    "age_in_year",
    "height_in_inch",
    "length_in_inch_side_0",
    "length_in_inch_side_1",
    "area_under_cow_in_sq_inch_side_0",
    "area_under_cow_in_sq_inch_side_1",
]

combined_sides_features_df[second_iteration_features]


# #### Define preprocessor

def create_second_iteration_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "age_in_year",
        "height_in_inch",
        "length_in_inch_side_0",
        "length_in_inch_side_1",
        "area_under_cow_in_sq_inch_side_0",
        "area_under_cow_in_sq_inch_side_1"
    ]
    categorical_features = ["sex", "color", "breed"]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(drop="first"))])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


# #### Score

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(400,), max_iter=1000, random_state=42),
}

second_iteration_preprocessor = create_second_iteration_preprocessor()

second_iteration_scores = pd.DataFrame.from_dict(
    {
        model_name: cross_validate_pipeline(
            X=combined_sides_features_df[second_iteration_features],
            y=combined_sides_features_df[target],
            pipeline=Pipeline(
                steps=[
                    ("preprocessor", second_iteration_preprocessor),
                    ("regressor", model)
                ]
            ),
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
        )
        for model_name, model in models.items()
    },
    orient="index",
)
second_iteration_scores

# <br>




