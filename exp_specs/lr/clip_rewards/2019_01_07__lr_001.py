from mrunner.experiment import Experiment
import sys, os
sys.path.append(os.path.join(os.getcwd(), 'exp_utils'))
from spec_utils import get_git_head_info, get_combinations
# It might be a good practice to not change specification files if run
# successfully, to keep convenient history of experiments. When you want to run
# the same experiment with different hyper-parameters, just copy it.
# Starting name with (approximate) date of run is also helpful.

def create_experiment_for_spec(parameters):
    script = 'baselines/acktr/run_atari_training.py'
    # this will be also displayed in jobs on prometheus
    name = 'lr=0.01, clip_rewards, sil-params, lkryston'
    project_name = "sil-montezuma"
    python_path = '.:exp_utils:some/other/utils/path'
    paths_to_dump = ''  # e.g. 'plgrid tensor2tensor', do we need it?
    tags = 'lkryston sil-params clip_rewards lr'.split(' ')
    parameters['git_head'] = get_git_head_info()
    return Experiment(project=project_name, name=name, script=script,
                      parameters=parameters, python_path=python_path,
                      paths_to_dump=paths_to_dump, tags=tags,
                      time='2-0'  # days-hours
                      )

# Set params_configurations, eg. as combinations of grid.
# params are also good place for e.g. output path, or git hash
params_grid = dict(
    death_penalty=[0.0],
    ent_coef=[0.01],
    env=['MontezumaRevengeNoFrameskip-v4'],
    exp_adv_est=['critic'],
    exp_max_score=[30000],
    expert_nbatch=[512],
    frame_stack=[4],
    gamma=[0.995],
    limit_len=[20000],
    limit_penalty=[0.0],
    load_model=[None],
    lr=[0.01],
    lrschedule=['constant'],
    max_grad_norm=[0.5],
    nsteps=[5],
    num_env=[16],
    random_state_reset=[False],
    episode_life=[True],
    clip_rewards=[True],
    sticky_action=[False],
    step_penalty=[0.0],
    use_n_trajectories=[-1],
    vf_coef=[0.5],
    sil_update=[4],
    sil_beta=[0.1]
)
params_configurations = get_combinations(params_grid)


def spec():
    experiments = [create_experiment_for_spec(params)
                   for params in params_configurations]
    return experiments