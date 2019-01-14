import joblib
import numpy as np
import tensorflow as tf
import time

from neptune import ChannelType

from baselines import logger
from baselines.acktr import kfac
from baselines.acktr.utils import Scheduler, find_trainable_variables
from baselines.acktr.utils import cat_entropy, mse
from baselines.acktr.utils import discount_with_dones
from baselines.common import set_global_seeds, explained_variance
from baselines.common.self_imitation import SelfImitation
from baselines.acktr.utils import EpisodeStats
from baselines.video.video_runners import easy_video


class Runner(object):
    def __init__(self, env, model, nsteps=5, gamma=0.99):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc)

        self.obs = np.zeros((nenv, nh, nw, nc), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_raw_rewards = [],[],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, raw_rewards, dones, _ = self.env.step(actions)
            rewards = np.sign(raw_rewards)
            self.states = states
            self.dones = dones
            if hasattr(self.model, 'sil'):
                self.model.sil.step(self.obs, actions, raw_rewards, dones)
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0

            self.obs = obs
            mb_rewards.append(rewards)
            mb_raw_rewards.append(raw_rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_raw_rewards = np.asarray(mb_raw_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_raw_rewards = mb_raw_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, mb_raw_rewards


class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs,
                 expert_nbatch,
                 total_timesteps,
                 nprocs=32, nsteps=20,
                 ent_coef=0.01,
                 vf_coef=0.5, vf_fisher_coef=1.0,
                 lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear',
                 sil_update=4, sil_beta=0.0):

        # create tf stuff
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=nprocs,
                                inter_op_parallelism_threads=nprocs)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)

        # the actual model
        nact = ac_space.n
        nbatch = nenvs * nsteps
        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
        eval_step_model = policy(sess, ob_space, ac_space, 1, 1, reuse=True)
        train_model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, reuse=True)
        sil_model = policy(sess, ob_space, ac_space, expert_nbatch, nsteps, reuse=True)

        logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        self.logits = logits = train_model.pi

        ## training loss
        pg_loss = tf.reduce_mean(ADV*logpac)
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        pg_loss = pg_loss - ent_coef * entropy
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        train_loss = pg_loss + vf_coef * vf_loss
        value_avg = tf.reduce_mean(train_model.vf)

        self.check = check = tf.add_check_numerics_ops()

        ## Fisher loss construction
        pg_fisher_loss = -tf.reduce_mean(logpac)
        sample_net = train_model.vf + tf.random_normal(tf.shape(train_model.vf))
        vf_fisher_loss = - vf_fisher_coef * tf.reduce_mean(tf.pow(train_model.vf - tf.stop_gradient(sample_net), 2))
        joint_fisher_loss = pg_fisher_loss + vf_fisher_loss

        params = find_trainable_variables("model")

        self.grads_check = grads = tf.gradients(train_loss, params)

        with tf.device('/gpu:0'):
            self.optim = optim = kfac.KfacOptimizer(
                learning_rate=LR, clip_kl=kfac_clip,
                momentum=0.9, kfac_update=1, epsilon=0.01,
                stats_decay=0.99, async=1, cold_iter=20, max_grad_norm=max_grad_norm
            )

            # why is this unused?
            update_stats_op = optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
            train_op, q_runner = optim.apply_gradients(list(zip(grads,params)))
        self.q_runner = q_runner
        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {
                train_model.X:obs,
                A:actions,
                ADV:advs,
                R:rewards,
                LR:cur_lr
            }

            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            policy_loss, value_loss, policy_entropy, v_avg, _, grads_to_check = sess.run(
                [pg_loss, vf_loss, entropy, value_avg, train_op, grads],
                td_map
            )

            for grad in grads_to_check:
                if np.isnan(grad).any():
                    print("ojojoj grad is nan")

            return policy_loss, value_loss, policy_entropy, v_avg

        self.sil = SelfImitation(sil_model.X, sil_model.vf,
                sil_model.entropy, sil_model.value, sil_model.neg_log_prob,
                ac_space, np.sign, n_env=nenvs, batch_size=expert_nbatch, n_update=sil_update, beta=sil_beta)
        self.sil.build_train_op(params, optim, LR, max_grad_norm=max_grad_norm)

        def sil_train():
            cur_lr = lr.value()
            return self.sil.train(sess, cur_lr)

        def save(save_path):
            print("Writing model to {}".format(save_path))
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        def eval_step(obs, eval_type):
            td_map = {eval_step_model.X: [obs]}
            logits = sess.run(eval_step_model.pi, td_map)[0]
            if eval_type == 'argmax':
                act = logits.argmax()
                if np.random.rand() < 0.01:
                    act = ac_space.sample()
                return act
            elif eval_type == 'prob':
                # probs = func(s[None, :, :, :])[0][0]
                x = logits
                e_x = np.exp(x - np.max(x))
                probs = e_x / e_x.sum(axis=0)
                act = np.random.choice(range(probs.shape[-1]), 1, p=probs)[0]
                return act
            else:
                raise ValueError("Unknown eval type {}".format(eval_type))

        self.model = step_model
        self.model2 = train_model
        self.vf_fisher = vf_fisher_loss
        self.pg_fisher = pg_fisher_loss
        self.joint_fisher = joint_fisher_loss
        self.params = params
        self.train = train
        self.save = save
        self.load = load
        self.train_model = train_model
        self.sil_train = sil_train
        self.step_model = step_model
        self.eval_step = eval_step
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=sess)
        tf.local_variables_initializer().run(session=sess)


def learn(policy, env, seed, ctx, params,
          expert_nbatch,
          exp_adv_est='reward',
          load_model=None,
          total_timesteps=int(40e6),
          gamma=0.99,
          nprocs=32, nsteps=20,
          ent_coef=0.01,
          vf_coef=0.5, vf_fisher_coef=1.0,
          lr=0.25, max_grad_norm=0.5,
          kfac_clip=0.001, lrschedule='linear', sil_update=4, sil_beta=0.0, video_interval=3600):

    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space

    model = Model(
        policy, ob_space, ac_space, nenvs,
        expert_nbatch,
        total_timesteps,
        nprocs=nprocs, nsteps=nsteps, ent_coef=ent_coef,
        vf_coef=vf_coef, vf_fisher_coef=vf_fisher_coef,
        lr=lr, max_grad_norm=max_grad_norm,
        kfac_clip=kfac_clip, lrschedule=lrschedule,
        sil_update=sil_update, sil_beta=sil_beta,
    )

    if load_model is not None and load_model != 'None':
        print("Loading model {}".format(load_model))
        model.load(load_model)

    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

    episode_stats = EpisodeStats(nsteps, nenvs)
    nbatch = nenvs*nsteps
    tstart = time.time()

    coord = tf.train.Coordinator()
    enqueue_threads = model.q_runner.create_threads(model.sess, coord=coord, start=True)

    t_last_update = tstart
    t_last_vid = 0

    mframes_channel = ctx.create_channel('mframes', channel_type=ChannelType.NUMERIC)
    fps_channel = ctx.create_channel('fps', channel_type=ChannelType.NUMERIC)
    policy_entropy_channel = ctx.create_channel('policy_entropy', channel_type=ChannelType.NUMERIC)
    policy_loss_channel = ctx.create_channel('policy_loss', channel_type=ChannelType.NUMERIC)
    value_loss_channel = ctx.create_channel('value_loss', channel_type=ChannelType.NUMERIC)
    explained_variance_channel = ctx.create_channel('explained_variance', channel_type=ChannelType.NUMERIC)
    episode_reward_channel = ctx.create_channel('episode_reward', channel_type=ChannelType.NUMERIC)
    sil_best_reward_channel = ctx.create_channel('sil_best_reward', channel_type=ChannelType.NUMERIC)
    sil_num_episodes_channel = ctx.create_channel('sil_num_episodes', channel_type=ChannelType.NUMERIC)
    sil_valid_samples_channel = ctx.create_channel('sil_valid_samples', channel_type=ChannelType.NUMERIC)
    sil_steps_channel = ctx.create_channel('sil_steps', channel_type=ChannelType.NUMERIC)

    update = 0


    while True:
        update += 1
        obs, states, rewards, masks, actions, values, raw_rewards = runner.run()
        episode_stats.feed(raw_rewards, masks)
        policy_loss, value_loss, policy_entropy, v_avg = model.train(obs, states, rewards, masks, actions, values)
        sil_loss, sil_adv, sil_samples, sil_nlogp = model.sil_train()
        model.old_obs = obs

        now = time.time()

        if now - t_last_update > 10:
            nseconds = now - tstart
            nframes = update * nbatch

            fps = int(float(nframes)/nseconds)
            mframes = nframes / 1e6

            t_last_update = now
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("mframes", mframes)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("episode_reward", episode_stats.mean_reward())
            logger.record_tabular("best_episode_reward", float(model.sil.get_best_reward()))
            if sil_update > 0:
                logger.record_tabular("sil_num_episodes", float(model.sil.num_episodes()))
                logger.record_tabular("sil_valid_samples", float(sil_samples))
                logger.record_tabular("sil_steps", float(model.sil.num_steps()))

            logger.dump_tabular()

            mframes_channel.send(x=nframes, y=mframes)
            fps_channel.send(x=nframes, y=fps)
            policy_entropy_channel.send(x=nframes, y=float(policy_entropy))
            policy_loss_channel.send(x=nframes, y=float(policy_loss))
            value_loss_channel.send(x=nframes, y=float(value_loss))
            explained_variance_channel.send(x=nframes, y=float(ev))
            episode_reward_channel.send(x=nframes, y=episode_stats.mean_reward())
            sil_best_reward_channel.send(x=nframes, y=float(model.sil.get_best_reward()))
            sil_num_episodes_channel.send(x=nframes, y=float(model.sil.num_episodes()))
            sil_valid_samples_channel.send(x=nframes, y=float(sil_samples))
            sil_steps_channel.send(x=nframes, y=float(model.sil.num_steps()))

        if now - t_last_vid > video_interval:
            easy_video(model, params, 'prob')
            easy_video(model, params, 'argmax')
            t_last_vid = time.time()

    coord.request_stop()
    coord.join(enqueue_threads)
    env.close()
