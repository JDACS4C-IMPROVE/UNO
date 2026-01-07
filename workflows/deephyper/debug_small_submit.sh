#!/bin/bash -l
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug 
#PBS -A IMPROVE_Aim1
#PBS -l filesystems=home:grand:eagle

module use /soft/modulefiles
module load nvhpc-mixed craype-accel-nvidia80
module load conda
conda activate 

cd ${PBS_O_WORKDIR}

# MPI example w/ 4 MPI ranks per node spread evenly across cores
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4
NDEPTH=8
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"


export PYTHONPATH=/lus/eagle/projects/IMPROVE_Aim1/rjain/IMPROVE/
export IMPROVE_DATA_DIR=/lus/eagle/projects/IMPROVE_Aim1/rjain/UNO/csa_data

export MPICH_GPU_SUPPORT_ENABLED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
mpirun -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} python hpo_deephyper_subprocess.py