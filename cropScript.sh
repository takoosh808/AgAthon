#!/bin/bash
#SBATCH --partition=kamiak  # Partition/Queue to use
#SBATCH --job-name=CropJob  # Job name
#SBATCH --output=%x_%j.out  # Output file (stdout)
#SBATCH --error=%x_%j.err  # Error file (stderr)
#SBATCH --time=00:05:00  # Wall clock time limit (HH:MM:SS)
#SBATCH --nodes=1  # Number of nodes
#SBATCH --ntasks-per-node=1  # Number of tasks per node (1 process)
#SBATCH --gpu-task=1  # Number of cores per task (threads)

# Define virtual environment name
# VENV_NAME="myenv"

# Define python file to run
PYTHON_FILE="CropResidue.py"

# Activate virtual environment
# source ~/$VENV_NAME/bin/activate
# echo "Activated virtual environment: $VENV_NAME"

# Function to run the Python script
run_python_script() {
    echo "Running Python script: $PYTHON_FILE"
    log_message "Running Python script: $PYTHON_FILE"

    srun python3 "$PYTHON_FILE"

    # Check if Python script executed successfully
    if [[ $? -eq 0 ]]; then
        log_message "Python script executed successfully."
        echo "Python script executed successfully."
    else
        log_message "Error: Python script failed to execute."
        echo "Error: Python script failed to execute."
        deactivate
        exit 1
    fi
    # deactivate
    # echo "Deactivated virtual environment: $VENV_NAME"
    # log_message "Deactivated virtual environment: $VENV_NAME"

    echo "Completed job on node $HOSTNAME"
}

# Main execution flow

check_file

run_python_script
