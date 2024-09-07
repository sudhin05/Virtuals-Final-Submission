#!/bin/bash

# Define the Conda environment name
CONDA_ENV_NAME="trauma"

# Define the paths to the directories containing the Python scripts
PYTHON_SCRIPT_2_DIR=$(realpath "grand_finale/CLIP_VIRTUAL_LODA")
PYTHON_SCRIPT_1_DIR=$(realpath "grand_finale/ViTPose/demo")

# Define the names of the Python scripts
PYTHON_SCRIPT_1="final_pipeline.py"
PYTHON_SCRIPT_2="main.py"

ARGUMENT1="$1"

# Activate the Conda environment
echo "Activating Conda environment: $CONDA_ENV_NAME"
source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path if necessary
conda activate "$CONDA_ENV_NAME"

# Change to the directory of the first Python script and run it
echo "Changing to directory: $PYTHON_SCRIPT_1_DIR"
cd "$PYTHON_SCRIPT_1_DIR" || { echo "Failed to change directory to $PYTHON_SCRIPT_1_DIR"; exit 1; }
echo "Running Python script: $PYTHON_SCRIPT_1"
python3 "$PYTHON_SCRIPT_1" "$ARGUMENT1"

# Deactivate the Conda environment
echo "Deactivating Conda environment"
conda deactivate

# Change to the directory of the second Python script and run it
echo "Changing to directory: $PYTHON_SCRIPT_2_DIR"
cd "$PYTHON_SCRIPT_2_DIR" || { echo "Failed to change directory to $PYTHON_SCRIPT_2_DIR"; exit 1; }
echo "Running Python script: $PYTHON_SCRIPT_2"
python3 "$PYTHON_SCRIPT_2" --folder_path "$ARGUMENT1"
