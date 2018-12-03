# Experiments Specification

One Paragraph of project description goes here

## Hyperparameters

The following hyperparameters were selected for the experiments:
 * entropy coefficient - **ent_coef**,
 * self-expert coefficient - **exp_coef**,
 * self-expert batch size - **exp_nbatch**,
 * discount factor - **gamma**,
 * learning rate - **lr**,
 * number of steps per iteration - **nsteps**.

In many cases, it was trivally assumed that the hyperparameters are *independent*.

## Logged info

During the experiments different statistics were gathered, including: 
 * the best (clipped) reawrd from an episode - **best_episode_reward**,
 * avarage episode reawrd - **episode_reward** (not present in every experiment),
 * training accuracy of the self-expert - **expert_train_accuracy**,
 * policy loss fo the self-expert - **policy_expert_loss**,
 * value loss - **value_loss**,
 * policy loss - **policy_loss**,
 * number of _positive_ self-expert episodes - **sil_num_episodes**,
 * total number of self-expert samples in the buffer - **sil_steps**.

## Directory structure

The experiments are orgnized in the following manner:

```
exp_specs/
|--README.md
|--<hparam1>/
|  |--README.md
|  |--<hparam1_exp_spec1>.py
|  |--<hparam1_exp_spec2>.py
......
|  |--<hparam1_exp_specN>.py
|--<hparam2>/
|  |--README.md
|  |--<hparam2_exp_spec1>.py
|  |--<hparam2_exp_spec2>.py
......
|  |--<hparam2_exp_specN>.py
...
|--<hparamN>/
|  |--README.md
|  |--<hparamN_exp_spec1>.py
|  |--<hparamN_exp_spec2>.py
......
|  |--<hparamN_exp_specN>.py
|--|--misc/
   |--README.md
   |--<misc_exp_spec1>.py
   |--<misc_exp_spec2>.py
   ...
   |--<misc_exp_specN>.py
```

where <hparamX> is an abbravation of the given hyperparameter and misc/ corresponds to _miscellaneous_ experiments. Each directory contains a README.md file with links to the experiments on the Neptune platform. The <hparamX_exp_specY>py file are of the form:

```
<YYYY_MM_DD_of_the_experiment>__<abbreviation_of_the_hyperparameter>_<value_of_the_hyperparameter>.py
e.g. '2018_12-01__gamma_099.py' corresponds to the experiment performed on 2918-12-01 with discount factor
y = 0.99
