## Reference
- [Distributional Soft Actor-Critic (DSAC)](https://arxiv.org/abs/2001.02811)


## Installation
```bash
# clone DSAC2.0 repository
git clone 
cd Distributional-Soft-Actor-Critic-2.0
# create conda environment
conda env create -f gops_environment.yml
conda activate gops
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
#Train an pendulum task
python main.py
#Train a humanoid task
python dsac_mlp_humanoidconti_offserial.py

```
After training, the results will be stored in the "results" folder.


## Acknowledgment
We would like to thank all members in Intelligent Driving Laboratory (iDLab), School of Vehicle and Mobility, Tsinghua University for making excellent contributions and providing helpful advices for DSAC 2.0.
