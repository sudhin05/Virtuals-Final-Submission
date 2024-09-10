PYTHON_SCRIPT_2="hr.py"

ARGUMENT1="$1"

# Replace this with the actual path to your global Python interpreter
GLOBAL_PYTHON_PATH="/bin/python3"  # Example path; replace with the correct one

echo "Changing to directory: $PYTHON_SCRIPT_2_DIR"
cd "$PYTHON_SCRIPT_2_DIR" || { echo "Failed to change directory to $PYTHON_SCRIPT_2_DIR"; exit 1; }
echo "Running Python script: $PYTHON_SCRIPT_2"
"$GLOBAL_PYTHON_PATH" "$PYTHON_SCRIPT_2" --root_path "$ARGUMENT1"
