# Distributed convex optimization using Over-the-air function computation technology

## Use following procedure to run the algorithm on a High Performance Computer (HPC) with SLURM cluster management and Singularity container

0. Build Singularity container using -> container.def

1. Write simulation configurations in file -> realdata_simulations.csv

2. Run for config ID x ->
    > $ sbatch DistOptOTAC.py x

3. After finish, simulations results will be saved in the `Results` folder.