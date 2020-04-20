#!/bin/sh

# You can control the resources and scheduling with '#SBATCH' settings(updating to commit)
# (see 'man sbatch' for more information on setting these parameters)

# The default partition is the 'general' partition
#SBATCH --partition=general

# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=long

# The default run (wall-clock) time is 1 minute
#SBATCH --time=50:00:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default number of CPUs per task is 1 (note: CPUs are always allocated per 2)
#SBATCH --cpus-per-task=2

# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=32768

# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --mail-type=END
#SBATCH --gres=gpu:2
# Your job commands go below here

# Uncomment these lines when your job requires this software
#module use /opt/insy/modulefiles
#module load cuda/10.0 cudnn/10.0-7.3.0.29
#module load cuda/9.0 cudnn/9.0-7.4.2.24


#module load matlab/R2018b

# Complex or heavy commands should be started with 'srun' (see 'man srun' for more information)
python3 ColorL7baseConv.py $*
# Use this simple command to check that your sbatch settings are working (verify the resources allocated in the usage statistics)
