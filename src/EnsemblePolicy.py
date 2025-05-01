import numpy as np
from scipy import stats
class EnsemblePolicy:
    def __init__(self, models, vote="majority"):
        self.models = models
        self.vote = vote
    def predict(self, obs, deterministic=True):
        actions = []
        for m in self.models:
            a, _ = m.predict(obs, deterministic=deterministic)
            actions.append(a)

        stacked = np.stack(actions, axis=0)
        maj = stats.mode(stacked, axis=0, keepdims=False).mode
        return maj, None
