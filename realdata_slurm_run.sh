#!/bin/bash
#SBATCH --job-name="realdata"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output="slurm_logs/slurm_out-%A_%a.out"
#SBATCH --error="slurm_logs/slurm_err-%A_%a.err"
#SBATCH --array=1-100

SIM_ID=$1
BO=$( echo "scale=2; $RANDOM*200/32767" | bc )

SEED=$((${SLURM_ARRAY_TASK_ID}))

CODE="DistOptOTAC.py"
CODE_DIR="/data/cluster/users/navya-xx/DistOptOTAC"
CODE_MNT="/mnt/simulations"

start=`date +%s.%N`
sleep $BO
singularity run --bind ${CODE_DIR}:${CODE_MNT} ./container.sif ${CODE} --rand-seed ${SEED} --sim-id ${SIM_ID} --debug
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l )
echo ${runtime}