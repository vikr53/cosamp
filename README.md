# Compressive Sampling (FIHT-WHT) for communication efficient Federated Learning

## Installation
Dependencies: `pytorch`, `tensorflow`, `numpy`, `mpi4py`, `ffht`
Install ffht by entering the "FFHT-master/" folder and running `python setup.py install`
mpi4py requires a working MPI Implementation (for instructions please check - [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html))

## Running the code
The code can be run using slurm on a cluster using the following command 
`sbatch run_cosamp.sbatch`

or locally using the following command
`mpiexec -n NUM_NODES python cosamp_resnet.py`