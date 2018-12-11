#!/usr/bin/env python3

import numpy as np
import os

from neptune import ChannelType

from baselines import logger
from baselines.acktr.acktr_disc import learn
from baselines.acktr.policies import CnnPolicy
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

from exp_utils.neptune_utils import get_configuration

import constants as cnst


def train(ctx):
    params = ctx.params
    policy_fn = CnnPolicy

    the_seed = np.random.randint(10000)
    print(80 * "SEED")
    print("Today's lucky seed is {}".format(the_seed))
    print(80 * "SEED")

    env = VecFrameStack(
        make_atari_env(
            env_id=params.env,
            num_env=params.num_env,
            seed=the_seed,
            limit_len=params.limit_len,
            limit_penalty=params.limit_penalty,
            death_penalty=params.death_penalty,
            step_penalty=params.step_penalty,
            random_state_reset=params.random_state_reset,
        ),
        params.frame_stack
    )

    learn(
        policy=policy_fn,
        env=env,
        seed=the_seed,
        ctx=ctx,
        params=params,
        expert_nbatch=params.expert_nbatch,
        exp_adv_est=params.exp_adv_est,
        load_model=params.load_model,
        gamma=params.gamma,
        nprocs=params.num_env,
        nsteps=params.nsteps,
        ent_coef=params.ent_coef,
        expert_coeff=params.exp_coeff,
        lr=params.lr,
        lrschedule=params.lrschedule,
        sil_update=params.sil_update,
        sil_beta=params.sil_beta,
    )

    env.close()


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['MRUNNER_UNDER_NEPTUNE'] = '1'

    ctx, exp_dir_path = get_configuration()
    logger.configure(dir=cnst.openai_logdir())

    debug_info = ctx.create_channel('debug info', channel_type=ChannelType.TEXT)
    debug_info.send('experiment path {}'.format(exp_dir_path))

    train(ctx)


if __name__ == '__main__':
    main()
