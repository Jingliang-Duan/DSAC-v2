from utils.sys_run import PolicyRunner
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
runner = PolicyRunner(
    log_policy_dir_list=["./results/DSAC_V2_gym_pendulum/240223-003213"],
    trained_policy_iteration_list=["8900_opt"],
    is_init_info=False,
    init_info={"init_state": [-1, 0.05, 0.05, 0, 0.1, 0.1]},
    save_render=True,
    legend_list=["DSAC_V2"],
    dt=0.01, # time interval between steps
)

runner.run()
