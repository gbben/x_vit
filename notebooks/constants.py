from pathlib import Path

repo_dir = Path(__file__).parent.parent
data_dir = repo_dir / "data"
models_dir = repo_dir / "models"
output_dir = repo_dir / "output"
image_regression_output_dir = output_dir / "image_regression"

# datasets
## cid
cid_dir = data_dir / "cid"
cid_images_dir = cid_dir / "images"
cid_yt_images_dir = cid_dir / "yt_images"
cid_dataset_csv = cid_dir / "dataset.csv"

cid_left_side_cow_masks_pq = cid_dir / "left_side_cow_masks.pq"
cid_left_side_dataset_pq = cid_dir / "left_side_dataset.pq"

cid_two_sides_df_pkl = cid_dir / "two_sides_df.pkl"
cid_two_sides_dataset_pq = cid_dir / "two_sides_dataset.pq"

## zone occupancy
zone_occupancy_data_dir = data_dir / "zone_occupancy"
zone_occupancy_frames_dir = zone_occupancy_data_dir / "frames"

# models
sam_model_file = models_dir / "FastSam-x.pt"
da2_metric_depth_model_file = models_dir / "depth_anything_v2_metric_hypersim_vitb.pth"
