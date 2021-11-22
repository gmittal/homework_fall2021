### Part 1
CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --use_rnd --unsupervised_exploration --exp_name q1_env1_rnd_easy

CUDA_VISIBLE_DEVICES=1 python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --unsupervised_exploration --exp_name q1_env1_random_easy

CUDA_VISIBLE_DEVICES=2 python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --exp_name q1_env1_rnd

CUDA_VISIBLE_DEVICES=3 python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --unsupervised_exploration --exp_name q1_env1_random

CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd --unsupervised_exploration --exp_name q1_env2_rnd

CUDA_VISIBLE_DEVICES=1 python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --unsupervised_exploration --exp_name q1_env2_random



### Part 1.2
CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --unsupervised_exploration --pc_expl --exp_name q1_alg_med

CUDA_VISIBLE_DEVICES=1 python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --unsupervised_exploration --pc_expl --exp_name q1_alg_hard

### Part 2
CUDA_VISIBLE_DEVICES=2 python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_dqn --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=3 python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exploit_rew_shift=1 --exploit_rew_scale=100

### Part 2.2
CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=5000 --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_5000  --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=1 python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=15000 --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_15000  --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=2 python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=5000 --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_5000  --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=3 python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=15000 --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_15000  --exploit_rew_shift=1 --exploit_rew_scale=100

### Part 2.3
CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.05 --exp_name q2_alpha0.05 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=1 python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exp_name q2_alpha0.1 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=2 python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.2 --exp_name q2_alpha0.2 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=3 python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.3 --exp_name q2_alpha0.3 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.5 --exp_name q2_alpha0.5 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=1 python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.7 --exp_name q2_alpha0.7 --exploit_rew_shift=1 --exploit_rew_scale=100

### Part 3
CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=0.0 --exp_name q3_medium_dqn --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=1 python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=1.0 --exp_name q3_medium_cql --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=2 python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=0.0 --exp_name q3_hard_dqn --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=3 python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd --num_exploration_steps=20000 --cql_alpha=1.0 --exp_name q3_hard_cql --exploit_rew_shift=1 --exploit_rew_scale=100

### Part 4
CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q4_awac_medium_unsupervised_lam0.1 --use_rnd --unsupervised_exploration --awac_lambda=0.1 --num_exploration_steps=20000 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q4_awac_medium_unsupervised_lam1 --use_rnd --unsupervised_exploration --awac_lambda=1 --num_exploration_steps=20000 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q4_awac_medium_unsupervised_lam2 --use_rnd --unsupervised_exploration --awac_lambda=2 --num_exploration_steps=20000 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q4_awac_medium_unsupervised_lam10 --use_rnd --unsupervised_exploration --awac_lambda=10 --num_exploration_steps=20000 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q4_awac_medium_unsupervised_lam20 --use_rnd --unsupervised_exploration --awac_lambda=20 --num_exploration_steps=20000 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=0 python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q4_awac_medium_unsupervised_lam50 --use_rnd --unsupervised_exploration --awac_lambda=50 --num_exploration_steps=20000 --exploit_rew_shift=1 --exploit_rew_scale=100



CUDA_VISIBLE_DEVICES=1 python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=0.1 --exp_name q4_awac_medium_supervised_lam0.1 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=1 python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=1 --exp_name q4_awac_medium_supervised_lam1 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=1 python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=2 --exp_name q4_awac_medium_supervised_lam2 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=1 python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=10 --exp_name q4_awac_medium_supervised_lam10 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=1 python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=20 --exp_name q4_awac_medium_supervised_lam20 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=1 python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=50 --exp_name q4_awac_medium_supervised_lam50 --exploit_rew_shift=1 --exploit_rew_scale=100



CUDA_VISIBLE_DEVICES=2 python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q4_awac_easy_unsupervised_lam0.1 --use_rnd --unsupervised_exploration --awac_lambda=0.1 --num_exploration_steps=20000 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=2 python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q4_awac_easy_unsupervised_lam1 --use_rnd --unsupervised_exploration --awac_lambda=1 --num_exploration_steps=20000 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=2 python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q4_awac_easy_unsupervised_lam2--use_rnd --unsupervised_exploration --awac_lambda=2 --num_exploration_steps=20000 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=2 python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q4_awac_easy_unsupervised_lam10 --use_rnd --unsupervised_exploration --awac_lambda=10 --num_exploration_steps=20000 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=2 python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q4_awac_easy_unsupervised_lam20 --use_rnd --unsupervised_exploration --awac_lambda=20 --num_exploration_steps=20000 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=2 python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q4_awac_easy_unsupervised_lam50 --use_rnd --unsupervised_exploration --awac_lambda=50 --num_exploration_steps=20000 --exploit_rew_shift=1 --exploit_rew_scale=100



CUDA_VISIBLE_DEVICES=3 python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=0.1 --exp_name q4_awac_easy_supervised_lam0.1 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=3 python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=1 --exp_name q4_awac_easy_supervised_lam1 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=3 python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=2 --exp_name q4_awac_easy_supervised_lam2 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=3 python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=10 --exp_name q4_awac_easy_supervised_lam10 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=3 python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=20 --exp_name q4_awac_easy_supervised_lam20 --exploit_rew_shift=1 --exploit_rew_scale=100

CUDA_VISIBLE_DEVICES=3 python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda=50 --exp_name q4_awac_easy_supervised_lam50 --exploit_rew_shift=1 --exploit_rew_scale=100