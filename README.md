# GATEPythonScript

Python script to use GATE mainly in PET dosimetry calculation.

Although I mainly use this scripy for PET, functions and classes in cluster.py and output.py can be generally used in others tasks.

## File Structure:

1. file_generate.py: Generate Macro files automatically based on inputs
2. cluster.py: **Easily Run GATE in diffenent processes simultaneously in one Computer.**
3. output.py: Analyse output files from GATE

## How to use:

First set up GATE environment and clone this project into your computer

> Generating Macro files is independent of the operating system and GATE environment.
>
> To run GATE directly using python, this project currently support
>
> - Windows with docker
>
> It should not be too difficult to implement in other circumstance, it's just I didn't use it such way.

1. Generate Macros: `file_generate.MacWriter`
2. Run in single process:
   1. Just run in command line or use `cluster.run_macro`
   2. Or, you can run in cluster to use more tools, just set `parallel=1`
3. Run in multiple processes:
   1. Split main.mac into several macros: `cluster.split_macro`
   2. Run them:  `cluster.run_cluster`
   3. Supervise the progress: `cluster.SimulationSupervisor` (Only support in cluster)
   4. When finished, merge all files: `cluster.ClusterMerger`，whose output is just like the output from one process
4. Analyse your outputs: `output`
