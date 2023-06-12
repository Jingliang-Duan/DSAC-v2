## Reference
- [Distributional Soft Actor-Critic (DSAC)](https://arxiv.org/abs/2001.02811)


## Requires
1. Windows 7 or greater or Linux.
2. Python 3.8.
3. The installation path must be in English.


## Installation
```bash
# Please make sure not to include Chinese characters in the installation path, as it may result in a failed execution.
# clone DSAC2.0 repository
git clone git@github.com:Jingliang-Duan/Distributional-Soft-Actor-Critic-2.0.git
cd Distributional-Soft-Actor-Critic-2.0
# create conda environment
conda env create -f DSAC2.0_environment.yml
conda activate DSAC2.0
# install DSAC2.0
pip install -e.
```


## Quick Start
These are two examples of running DSAC2.0 on two environments. 
Train the policy by running:
```bash
#Train a pendulum task
python main.py
#Train a humanoid task. To execute this file, Mujoco and Mujoco-py need to be installed first. 
python dsac_mlp_humanoidconti_offserial.py
```
After training, the results will be stored in the "Distributional-Soft-Actor-Critic-2.0/results" folder.


## Acknowledgment
We would like to thank all members in Intelligent Driving Laboratory (iDLab), School of Vehicle and Mobility, Tsinghua University for making excellent contributions and providing helpful advices for DSAC 2.0.
