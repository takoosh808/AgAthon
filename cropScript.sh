#!/bin/bash
#SBATCH --account=hackathon --reservation=hackathon_gpu --gres=gpu:1 # Partition/Queue to use
#SBATCH --job-name=CropJob8  # Job name
#SBATCH --output=%x_%j.out  # Output file (stdout)
#SBATCH --error=%x_%j.err  # Error file (stderr)
#SBATCH --time=00:05:00  # Wall clock time limit (HH:MM:SS)
#SBATCH --nodes=1  # Number of nodes
#SBATCH --ntasks-per-node=1  # Number of tasks per node (1 process)
#SBATCH --cpus-per-task=16  # Number of cores per task (threads)
#SBATCH --mem=64GB
#SBATCH --mail-user=benjamin.k.metzger@wsu.edu

# Define python file to run
PYTHON_FILE="CropResidue.py"


# Function to run the Python script
run_python_script() {
    echo "Running Python script: $PYTHON_FILE"
    log_message "Running Python script: $PYTHON_FILE"

    run "$PYTHON_FILE"

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

    echo "Completed job on node $HOSTNAME"
}

# Main execution flow
module load singularity
singularity exec --nv \
    /scratch/project/hackathon/data/CropResiduePredictionChallenge/my-tf-container.sif \ 
    run_python_script()