# -

```bash
# Convert EurekaWorld sim rollouts -> LeRobot dataset (DROID schema)
cd /home/huluwulu/Projects/eurekaworld/openpi

# (first time only)
GIT_LFS_SKIP_SMUDGE=1 uv sync

uv run examples/droid/convert_eurekaworld_sim_dataset_to_lerobot.py \
  --data_dir /home/huluwulu/Projects/eurekaworld/datasets/sim_openpi_droid_lift/20260105_122524 \
  --repo_id huluwulu/sim_openpi_droid_lift_20260105_122524

# Optional: override FPS (otherwise inferred from meta.json)
uv run examples/droid/convert_eurekaworld_sim_dataset_to_lerobot.py \
  --data_dir /home/huluwulu/Projects/eurekaworld/datasets/sim_openpi_droid_lift/20260105_122524 \
  --repo_id huluwulu/sim_openpi_droid_lift_20260105_122524 \
  --fps 25

# Output:
#   ./lerobot_datasets/<repo_id>

# If you want OpenPI training to find the dataset by repo_id, use:
#   export HF_LEROBOT_HOME=./lerobot_datasets

# Optional: write the dataset somewhere else
uv run examples/droid/convert_eurekaworld_sim_dataset_to_lerobot.py \
  --data_dir /home/huluwulu/Projects/eurekaworld/datasets/sim_openpi_droid_lift/20260105_122524 \
  --repo_id huluwulu/sim_openpi_droid_lift_20260105_122524 \
  --output_dir /some/other/path
```
