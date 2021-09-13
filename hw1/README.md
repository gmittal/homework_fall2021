## Setup

You can run this code on your own machine or on Google Colab. 

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](installation.md) for instructions.
2. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeleydeeprlcourse/homework_fall2021/blob/master/hw1/cs285/scripts/run_hw1.ipynb)

## Complete the code

Fill in sections marked with `TODO`. In particular, see
 - [infrastructure/rl_trainer.py](cs285/infrastructure/rl_trainer.py)
 - [policies/MLP_policy.py](cs285/policies/MLP_policy.py)
 - [infrastructure/replay_buffer.py](cs285/infrastructure/replay_buffer.py)
 - [infrastructure/utils.py](cs285/infrastructure/utils.py)
 - [infrastructure/pytorch_util.py](cs285/infrastructure/pytorch_util.py)

Look for sections maked with `HW1` to see how the edits you make will be used.
Some other files that you may find relevant
 - [scripts/run_hw1.py](cs285/scripts/run_hw1.py) (if running locally) or [scripts/run_hw1.ipynb](cs285/scripts/run_hw1.ipynb) (if running on Colab)
 - [agents/bc_agent.py](cs285/agents/bc_agent.py)

See the homework pdf for more details.

## Run the code

Tip: While debugging, you probably want to keep the flag `--video_log_freq -1` which will disable video logging and speed up the experiment. However, feel free to remove it to save videos of your awesome policy!

If running on Colab, adjust the `#@params` in the `Args` class according to the commmand line arguments above.

### Section 1 (Behavior Cloning)
To reproduce results from Section 1 (Ant-v2 and HalfCheetah-v2):

```
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
--expert_data cs285/expert_data/expert_data_Ant-v2.pkl \
--ep_len 1000 --eval_batch_size 10000 \
--video_log_freq -1


python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
--env_name HalfCheetah-v2 --exp_name bc_halfcheetah --n_iter 1 \
--expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl \
--ep_len 1000 --eval_batch_size 10000 \
--video_log_freq -1
```

To generate the numbers shown in the ablation for 1.3:
```
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
--env_name HalfCheetah-v2 --exp_name bc_halfcheetah --n_iter 1 \
--expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl \
--ep_len 1000 --eval_batch_size 10000 \
--num_agent_train_steps_per_iter 125 \
--video_log_freq -1

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
--env_name HalfCheetah-v2 --exp_name bc_halfcheetah --n_iter 1 \
--expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl \
--ep_len 1000 --eval_batch_size 10000 \
--num_agent_train_steps_per_iter 250 \
--video_log_freq -1

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
--env_name HalfCheetah-v2 --exp_name bc_halfcheetah --n_iter 1 \
--expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl \
--ep_len 1000 --eval_batch_size 10000 \
--num_agent_train_steps_per_iter 500 \
--video_log_freq -1

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
--env_name HalfCheetah-v2 --exp_name bc_halfcheetah --n_iter 1 \
--expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl \
--ep_len 1000 --eval_batch_size 10000 \
--num_agent_train_steps_per_iter 1000 \
--video_log_freq -1

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
--env_name HalfCheetah-v2 --exp_name bc_halfcheetah --n_iter 1 \
--expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl \
--ep_len 1000 --eval_batch_size 10000 \
--num_agent_train_steps_per_iter 2000 \
--video_log_freq -1

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
--env_name HalfCheetah-v2 --exp_name bc_halfcheetah --n_iter 1 \
--expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl \
--ep_len 1000 --eval_batch_size 10000 \
--num_agent_train_steps_per_iter 4000 \
--video_log_freq -1

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
--env_name HalfCheetah-v2 --exp_name bc_halfcheetah --n_iter 1 \
--expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl \
--ep_len 1000 --eval_batch_size 10000 \
--num_agent_train_steps_per_iter 8000 \
--video_log_freq -1

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
--env_name HalfCheetah-v2 --exp_name bc_halfcheetah --n_iter 1 \
--expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl \
--ep_len 1000 --eval_batch_size 10000 \
--num_agent_train_steps_per_iter 16000 \
--video_log_freq -1

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
--env_name HalfCheetah-v2 --exp_name bc_halfcheetah --n_iter 1 \
--expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl \
--ep_len 1000 --eval_batch_size 10000 \
--num_agent_train_steps_per_iter 32000 \
--video_log_freq -1
```

### Section 2 (DAgger)
To reproduce results in Section 2 (for Ant-v2 and HalfCheetah-v2) run:

```
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name dagger_ant --n_iter 10 \
--do_dagger --expert_data cs285/expert_data/expert_data_Ant-v2.pkl \
--ep_len 1000 --eval_batch_size 10000 \
--video_log_freq -1

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
--env_name HalfCheetah-v2 --exp_name dagger_halfcheetah --n_iter 10 \
--do_dagger --expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl \
--ep_len 1000 --eval_batch_size 10000 \
--video_log_freq -1
```

## Visualization the saved tensorboard event file:

You can visualize your runs using tensorboard:
```
tensorboard --logdir data
```

You will see scalar summaries as well as videos of your trained policies (in the 'images' tab).

You can choose to visualize specific runs with a comma-separated list:
```
tensorboard --logdir data/run1,data/run2,data/run3...
```

If running on Colab, you will be using the `%tensorboard` [line magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html) to do the same thing; see the [notebook](cs285/scripts/run_hw1.ipynb) for more details.

