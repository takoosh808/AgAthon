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
module load singularity
module load anaconda3

export TF_ENABLE_ONEDNN_OPTS=0

source activate cropenv
echo "Running Python script: $PYTHON_FILE"


singularity exec --nv /scratch/project/hackathon/data/CropResiduePredictionChallenge/my-tf-container.sif \
    python3 "$PYTHON_FILE"
fi

echo "Completed job on node $HOSTNAME"
source deactivate