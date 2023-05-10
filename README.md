## Reference
- [Distributional Soft Actor-Critic (DSAC)](https://arxiv.org/abs/2001.02811)


## Installation
```bash
# clone DSAC2.0 repository
git clone 
cd Distributional-Soft-Actor-Critic-2.0
# create conda environment
conda env create -f gops_environment.yml
conda activate DSAC2.0
# install DSAC2.0
pip install -e.
```


## Requires
1. Windows 7 or greater or Linux.
2. Python 3.8.
3. The installation path must be in English.


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
