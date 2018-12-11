class ExpertRunner(object):
    def __init__(self, env, model):
        self.env = env
        self.model = model

    def run(self):
        """
        Return a batch of
        - observations
        - values according to the freshest value function estimate
        - expert actions
        """
        obs, actions, returns, _, idxes = self.model.sil.sample_batch(self.model.sil.batch_size)
        values = self.model.expert_train_model.value(obs)
        return obs, actions, returns, values, idxes