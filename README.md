## Requirements
 - deall.ii 8.5.0 - https://www.dealii.org/ - with support for
     - PETSc 3.6.0
     - METIS 5.1.0
 - some custom libraries - https://github.com/zerpiko/libraries
 - GNU compiler 4.8.5
 - Intel MPI 2016 (20160601)
 - GMSH 2.7.0 (to visualize mesh files)

Other compilers and libraries might work but I haven't tested them.

## Instructions
The program expects only one argument, the name of the file with the input parameters.
To execute:

```shell
./mycode input.prm
```

The program requires an `output` directory in the same location where the program is
executed.

## Mesh files
Current mesh:

 - trl_refined_in_2d.msh : 2706 vertices, 3038 elements

This mesh can be further refined within the program if needed.

| whole domain | zoom to pipes section |
|----------------------|--------------------|
| ![whole domain](./input/figures/trl_mesh_in_2d.png) | ![zoom to pipes section](./input/figures/trl_mesh_in_2d_pipes.png) |

## Simulation periods
The simulation is split in **preheating** steps. Each step has different duration.
Meteorological data is reused year after year, this is important specially in the 1st 
step (8 years).

- preheating step 1. 70079 hourly time steps (8 years). 
  Purpose: reach a stationary state in the domain based on given meteorological and
  boundary conditions.


## Visualizing results
The code will generate the files solution-some-string-TTTT.NNN.vtu, where TTTT is the
timestep number (starting from 1) and NNN is the process rank (starting from 0). These
files contain the locally owned cells for the timestep and processor. The 
solution-some-string-TTTT.pvtu is the paraview record for timestep TTTT. Finally, the
file solution.pvd is a special record only supported by ParaView that references all
time steps. So in ParaView, only solution.pvd needs to be opened. See dealli
documentation for [step 18](https://www.dealii.org/current/doxygen/deal.II/step_18.html) 




