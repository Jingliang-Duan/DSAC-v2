## Reference
- [Distributional Soft Actor-Critic (DSAC)](https://arxiv.org/abs/2001.02811)


## Installation
1. Windows 7 or greater or Linux.
2. Python 3.6 or greater (GOPS V1.0 precompiled Simulink models use Python 3.8). We recommend using Python 3.8.
3. (Optional) Matlab/Simulink 2018a or greater.
4. The installation path must be in English.


## Quick Start
This is an example of running finite-horizon Approximate Dynamic Programming (FHADP) on inverted double pendulum environment. 
Train the policy by running:
```bash
python example_train/fhadp/fhadp_mlp_idpendulum_serial.py
```
After training, test the policy by running:
```bash
python example_run/run_idp_fhadp.py
```


## Acknowledgment
We would like to thank all members in Intelligent Driving Laboratory (iDLab), School of Vehicle and Mobility, Tsinghua University for making excellent contributions and providing helpful advices for DSAC 2.0.
