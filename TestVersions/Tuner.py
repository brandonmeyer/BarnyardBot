from ray import tune
from BarnyardBot.py import *
from ray.tune import choice, loguniform
import ray.rllib.agents.ppo as ppo
import gym

num_workers = 4
use_gpu = False
results_base_dir = os.path.join(expanduser("~"), "tune_dir")
ray.shutdown()
ray.init(num_cpus=num_workers, num_gpus=int(use_gpu))

tune.run(
     ppo.PPOTrainer,
     name="BarnyardBot_Tuning",
     metric="episode_reward_mean",
     loggers=[get_trainer_logger_creator(
             base_dir=results_base_dir,
             experiment_name="my_mujoco",
             should_log_result_fn=lambda result: result["training_iteration"] % 1 == 0,
             delete_hist_stats=False
                                    )],
     config={
            'env': MujocoCustomEnv,
            'env_config': {},
            'framework': 'torch',
            'num_gpus': int(use_gpu),
            'num_gpus_per_worker': 0,
            'num_workers': num_workers-1,
            'simple_optimizer': True,
            # Should use a critic as a baseline (otherwise don't use value baseline;
            # required for using GAE).
            "use_critic": True,
            # If true, use the Generalized Advantage Estimator (GAE)
            # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
            "use_gae": True,
            # The GAE (lambda) parameter.
            "lambda": choice([1.0, 0.95, 0.90]),
            # Initial coefficient for KL divergence.
            "kl_coeff": choice([0.1, 0.2, 0.3]),
            # Size of batches collected from each worker.
            "rollout_fragment_length": 200,
            # Number of timesteps collected for each SGD round. This defines the size
            # of each SGD epoch.
            "train_batch_size": choice([1024, 4096]),
            # Total SGD batch size across all devices for SGD. This defines the
            # minibatch size within each epoch.
            "sgd_minibatch_size": choice([64, 128]),
            # Whether to shuffle sequences in the batch when training (recommended).
            "shuffle_sequences": True,
            # Number of SGD iterations in each outer loop (i.e., number of epochs to
            # execute per train batch).
            "num_sgd_iter": choice([30, 10, 5]),
            # Stepsize of SGD.
            "lr": loguniform(5e-6, 5e-3),
            # Learning rate schedule.
            "lr_schedule": None,
            # Coefficient of the value function loss. IMPORTANT: you must tune this if
            # you set vf_share_layers=True inside your model's config.
            "vf_loss_coeff": 1.0,
            "model": {
            # Share layers for value function. If you set this to True, it's
            # important to tune vf_loss_coeff.
            "vf_share_layers": False,
            },
            # Coefficient of the entropy regularizer.
            "entropy_coeff": choice([0.0, 0.01, 0.001]),
            # Decay schedule for the entropy regularizer.
            "entropy_coeff_schedule": None,
            # PPO clip parameter.
            "clip_param": choice([0.2, 0.3, 0.4]),
            # Clip param for the value function. Note that this is sensitive to the
            # scale of the rewards. If your expected V is large, increase this.
            "vf_clip_param": choice([50.0, 100.0, 500.0]),
            # If specified, clip the global norm of gradients by this amount.
            "grad_clip": None,
            # Target value for KL divergence.
            "kl_target": choice([0.001, 0.01, 0.1]),
            # Whether to rollout "complete_episodes" or "truncate_episodes".
            "batch_mode": "truncate_episodes",
            # Which observation filter to apply to the observation.
            "observation_filter": "NoFilter"

            # Deprecated keys:
            # Share layers for value function. If you set this to True, it's important
            # to tune vf_loss_coeff.
            # Use config.model.vf_share_layers instead.
            #"vf_share_layers": "DEPRECATED_VALUE"
            },
     num_samples=20000,
     stop={
     'training_iteration': 200
     }
 )