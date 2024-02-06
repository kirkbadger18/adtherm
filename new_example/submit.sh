#!/usr/bin/env bash
#SBATCH --job-name=AdTherm_test
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem=96G
#SBATCH --account=cfgoldsm-condo

module load anaconda/2023.09-0-7nso27y
source /oscar/runtime/opt/anaconda/2023.03-1/etc/profile.d/conda.sh
conda activate /users/kbadger1/.conda/envs/math

module load hpcx-mpi
module load quantum-espresso-mpi 

export PYTHONPATH=/gpfs/data/cfgoldsm/kirk/ase:$PYTHONPATH

#mpirun -np 48 /oscar/runtime/software/external/quantum-espresso/7.1/bin/pw.x
#mpirun /oscar/runtime/software/external/quantum-espresso/7.1/bin/pw.x < espresso.pwi > espresso.pwo
python3 methyl_geom_gen.py
