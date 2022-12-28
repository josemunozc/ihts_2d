#!/bin/bash
#SBATCH -J ihts_2d
#SBATCH -o o.%x.%j
#SBATCH -e e.%x.%j
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=40
#SBATCH --time=48:00:00
#SBATCH --partition=compute
#SBATCH --account=scw1001

set -eu

module use ~/bench/modules
module purge
module load my-dealii/8.5.0
module list

root=${SLURM_SUBMIT_DIR}

WORKDIR=/scratch/$USER/ihts_2d.${SLURM_JOB_ID}
rm -rf ${WORKDIR}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

cp $root/build/mycode .
cp $root/input.prm .
mkdir input
mkdir input/meshes
mkdir input/met_data
mkdir output
cp $root/input/meshes/trl_mesh_in_2d.msh input/meshes
cp -r $root/input/met_data/trl input/met_data

mpirun -np ${SLURM_NTASKS} ./mycode input.prm
