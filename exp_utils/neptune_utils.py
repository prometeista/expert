import argparse
import os
from deepsense import neptune


def is_neptune_online():
  # I wouldn't be suprised if this would depend on neptune version
  return 'NEPTUNE_ONLINE_CONTEXT' in os.environ


def get_configuration():
  if is_neptune_online():
    # running under neptune
    ctx = neptune.Context()
    # I can't find storage path in Neptune 2 context
    # exp_dir_path = ctx.storage_url - this was used in neptune 1.6
    exp_dir_path = os.getcwd()
  else:
    # local run
    parser = argparse.ArgumentParser(description='Debug run.')
    parser.add_argument('--exp_spec', type=str)
    parser.add_argument("--exp_dir_path", default='/tmp')
    commandline_args = parser.parse_args()
    if commandline_args.exp_spec != None:
      vars = {}
      exec(open(commandline_args.exp_spec).read(), vars)
      spec_func = vars['spec']
      # take first experiment (params configuration)
      experiment = spec_func()[0]
      params = experiment.parameters
    else:
      params = {}
    # create offline context
    ctx = neptune.Context(offline_parameters=params)
    exp_dir_path = commandline_args.exp_dir_path
  return ctx, exp_dir_path
