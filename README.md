# DMD_over-relaxation
Dynamic Mode Decomposition (DMD) meets over-relaxation
This is a Python scripts that takes as input a matrix of solution updates (solution update snapshots at different time-steps) and outputs a solution update vector that you can use to over-relax your solution towards steady-state.
At this time, the script can handle snapshots matrix, with each snapshot taken from equally spaced time-steps.

The code uses Dynamic Mode Decomposition to analayze solution update matrix, find the slow converging modes, and outputs an update vector to eliminate those slow converging modes.
You can pass a flag to the script to output the solution modes and the corresponding time-dynamics of the solution update modes.

Arguments of the script (in order): 
- address to the vector file
- number of modes to eliminate,
- time-step,
- flag to calculate solution modes and time-dynamics (True or False)
