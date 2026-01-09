# EurekaWorld → OpenPI dataset conversion (LeRobot / DROID)

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

# Output (default):
#   ./lerobot_datasets/<repo_id>
#   (i.e., /home/huluwulu/Projects/eurekaworld/openpi/lerobot_datasets/<repo_id>)
#
# Note: OpenPI sets `HF_LEROBOT_HOME=./lerobot_datasets` at import time (unless you already set it),
# so training scripts will automatically find this dataset by repo_id.

# Optional: write the dataset under the shared datasets folder
uv run examples/droid/convert_eurekaworld_sim_dataset_to_lerobot.py \
  --data_dir /home/huluwulu/Projects/eurekaworld/datasets/sim_openpi_droid_lift/20260105_122524 \
  --repo_id huluwulu/sim_openpi_droid_lift_20260105_122524 \
  --output_dir /home/huluwulu/Projects/eurekaworld/datasets/lerobot_datasets

uv run examples/droid/convert_eurekaworld_sim_dataset_to_lerobot.py \
  --data_dir /home/huluwulu/Projects/eurekaworld/datasets/sim_openpi_droid_lift/20260106_181901 \
  --repo_id huluwulu/sim_openpi_droid_lift_wo_R_20260106_181901 \
  --output_dir /home/huluwulu/Projects/eurekaworld/datasets/lerobot_datasets

# Then set HF_LEROBOT_HOME before training:
#   export HF_LEROBOT_HOME=/home/huluwulu/Projects/eurekaworld/datasets/lerobot_datasets
```

Finetuning Script
```bash
cd /home/huluwulu/Projects/eurekaworld/openpi
HF_LEROBOT_HOME=/home/huluwulu/Projects/eurekaworld/datasets/lerobot_datasets \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_droid_lora_finetune \
  --exp-name sim_openpi_droid_lift_wo_R_20260106_181901_lora \
  --overwrite \
  --data.repo-id huluwulu/sim_openpi_droid_lift_wo_R_20260106_181901
```
