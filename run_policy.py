from utils.sys_run import PolicyRunner

runner = PolicyRunner(
    log_policy_dir_list=["./results/gym_pendulum/231026-223844"],
    trained_policy_iteration_list=["11800_opt"],
    is_init_info=False,
    init_info={"init_state": [-1, 0.05, 0.05, 0, 0.1, 0.1]},
    save_render=True,
    legend_list=["DSAC"],
    dt=0.01, # time interval between steps
)

runner.run()
