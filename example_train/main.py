import argparse
import os

os.environ["OMP_NUM_THREADS"] = "4"

from utils.initialization import create_alg, create_buffer, create_env
from training.trainer import create_trainer
from training.off_sampler import create_sampler
from training.evaluator import create_evaluator
from utils.init_args import init_args
from utils.tensorboard_setup import start_tensorboard, save_tb_to_csv

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="gym_pendulum")  #gym_pendulum can be replaced by other envs in the env_gym folder, such as gym_ant, gym_walker2d...
    parser.add_argument("--algorithm", type=str, default="DSAC")
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")
    ################################################
    # 1. Parameters for environment
    parser.add_argument(
        "--action_type", type=str, default="continu", help="Options: continu/discret"
    )
    parser.add_argument(
        "--is_render", type=bool, default=False, help="Draw environment animation"
    )
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training")
    parser.add_argument("--reward_scale", type=float, default=0.1)
    ################################################
    # 2.1 Parameters of value approximate function

    parser.add_argument("--value_func_name", type=str, default="ActionValueDistri")
    # Options: MLP/CNN/RNN/POLY/GAUSS
    parser.add_argument("--value_func_type", type=str, default="MLP")
    value_func_type = parser.parse_known_args()[0].value_func_type
    parser.add_argument("--value_hidden_sizes", type=list, default=[64, 64])
    # Hidden Layer Options: relu/gelu/elu/sigmoid/tanh
    parser.add_argument("--value_hidden_activation", type=str, default="relu")
    # Output Layer: linear
    parser.add_argument("--value_output_activation", type=str, default="linear")
    parser.add_argument("--value_min_log_std", type=int, default=-0.1)
    parser.add_argument("--value_max_log_std", type=int, default=4)

    # policy net
    # 2.2 Parameters of policy approximate function
    parser.add_argument("--policy_func_name", type=str, default="StochaPolicy")
    parser.add_argument("--policy_func_type", type=str, default="MLP")
    parser.add_argument(
        "--policy_act_distribution", type=str, default="TanhGaussDistribution"
    )
    parser.add_argument("--policy_hidden_sizes", type=list, default=[64, 64])
    parser.add_argument("--policy_hidden_activation", type=str, default="relu")
    parser.add_argument("--policy_output_activation", type=str, default="linear")
    parser.add_argument("--policy_min_log_std", type=int, default=-20)
    parser.add_argument("--policy_max_log_std", type=int, default=0.5)

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=1e-3)
    parser.add_argument("--policy_learning_rate", type=float, default=1e-3)
    parser.add_argument("--alpha_learning_rate", type=float, default=1e-3)
    # special parameter
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--auto_alpha", type=bool, default=True)
    parser.add_argument("--alpha", type=bool, default=0.2)
    parser.add_argument("--delay_update", type=int, default=2, help="")
    parser.add_argument("--TD_bound", type=float, default=10)
    parser.add_argument("--bound", default=True)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument("--trainer", type=str, default="off_serial_trainer")
    parser.add_argument("--max_iteration", type=int, default=8000)
    parser.add_argument("--ini_network_dir", type=str, default=None)
    trainer_type = parser.parse_known_args()[0].trainer
    # 4.1. Parameters for off_serial_trainer
    parser.add_argument("--buffer_name", type=str, default="replay_buffer")
    # Size of collected samples before training
    parser.add_argument("--buffer_warm_size", type=int, default=int(1e3))
    # Max size of reply buffer
    parser.add_argument("--buffer_max_size", type=int, default=int(1e5))
    # Batch size of replay samples from buffer
    parser.add_argument("--replay_batch_size", type=int, default=64)
    # Period of sampling
    parser.add_argument("--sample_interval", type=int, default=1)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=8)
    # Add noise to action for better exploration
    parser.add_argument("--noise_params", type=dict, default=None)
    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=5)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")
    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=5000)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=100)

    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)

    start_tensorboard(args["save_folder"])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    # Step 2: create sampler in trainer
    sampler = create_sampler(**args)
    # Step 3: create buffer in trainer
    buffer = create_buffer(**args)
    # Step 4: create evaluator in trainer
    evaluator = create_evaluator(**args)
    # Step 5: create trainer
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    # Start training ... ...
    trainer.train()
    print("Training is finished!")

    # Plot and save training figures
    save_tb_to_csv(args["save_folder"])
    print("Plot & Save are finished!")
