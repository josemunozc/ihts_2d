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
- preheating step 2. 5804 hourly time steps (9 months)
  Purpose: no difference in conditions with the previous period but it works to fill
  the gap until the next period with different conditions.
- preheating step 3. 2735 hourly time steps (about 3.7 months, from May 1st to Aug 23).
  Purpose: this covers the period when the insulation layer and pipe system were put in
  place in early May 2005 until the system was actually activated by the end of August
  2005. The soil thermal profile was impacted by the presence of the insulation layer
  as it wasn't allowed to gain heat at usual over the summer before the system was turned
  on.
- preheating step 4. 7964 time steps of 15 min (from 23/08/2005 to 14/11/2005).
  Purpose: 1st charging period. The system is activated to collect thermal energy from
  the pavement and transfer it to the volume under the insulation layer. There are two
  ways to activate the system: automatic and forced. In the former method the system
  is activated automatically if a certain criteria is meet (the average temperature
  difference between the collector and storage pipes). For the latter, the system is
  forced to be active in certain ranges of time (i.e. between noon and 10pm everyday).
- preheating step 5. 9500 time steps of 15 min (from 15/11/2005 to 20/02/2006).
  Purpose: 1st usage period. Thermal energy is transferred from the storage area to
  the collector pipes to warm up the pavement. System activation is similar to the
  description in preheating step 4.
- preheating step 6. 1559 hourly time steps (from 21/02/2006 to 28/04/2006).
  Purpose: During this period the system was stopped for repairs.
- preheating step 7. 


## Visualizing results
The code will generate the files solution-some-string-TTTT.NNN.vtu, where TTTT is the
timestep number (starting from 1) and NNN is the process rank (starting from 0). These
files contain the locally owned cells for the timestep and processor. The 
solution-some-string-TTTT.pvtu is the paraview record for timestep TTTT. Finally, the
file solution.pvd is a special record only supported by ParaView that references all
time steps. So in ParaView, only solution.pvd needs to be opened. See dealli
documentation for [step 18](https://www.dealii.org/current/doxygen/deal.II/step_18.html) 


## Building the code
Assuming the libraries listed in the Requirements section are available, it should be as simple as:

```
mkdir build
cd build
cmake ..
make
```

## Results

Job IDs reference SLURM IDs on the Hawk cluster.

| Job ID    | Description  | time step | Insulation | system active? |
|-----------|--------------|-----------|------------|----------------|
| 54986200  | preheating 1 | 3600      | False      | False          |
| 55042331  | preheating 2 | 3600      | False      | False          |
| 55042396  | preheating 3 | 3600      |  True      | False          |
| 55042415  | preheating 4 |  900      |  True      |  True          |
| 55044683  | preheating 5 |  900      |  True      |  True          |
| 55044754  | preheating 6 | 3600      |  True      | False          |
| 55044758  | preheating 7 |  900      |  True      |  True          |
| 55046962  | preheating 8 |  900      |  True      |  True          |