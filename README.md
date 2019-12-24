# Variational Recurrent Models for Solving Partially Observable Control Tasks
Codes for the study "Variational Recurrent Models for Solving Partially Observable Control Tasks", published as a conference paper at ICLR 2020 (https://openreview.net/forum?id=r1lL4a4tDB)

## Language: 
- Python3.6

## Library dependencies:
- PyTorch (1.1.0, CPU version)
- numpy (1.16.4)
- scipy (1.3.0)
- gym (0.12.5)
- roboschool (for the robotic control tasks) 
- docopt (0.6.2)

### To run an experiment:

```
python run_experiment.py run --env=PendulumV --steps=50000 --seed=0 --render
```
The program will run and save the results as .mat files at './data/' when finished (you can read it using scipy.io.loadmat in Python). Also, the trained agent will be saved as a PyTorch Module.

## List of available env_names (V: velocities only, P: no velocities):
- Pendulum
- PendulumV
- PendulumP
- CartPole
- CartPoleV
- CartPoleP
- Hopper
- HopperV
- HopperP
- Ant
- AntV
- AntP
- Walker2d
- Walker2dP
- Walker2dV
- Sequential

