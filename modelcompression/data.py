from utils import iterate_minibatches
from utils import provider_from_module

class Generator(object):

    def generate(self):
        raise NotImplementedError


class MiniBatchGenerator(object):

    def __init__(self,
                 X,
                 teacher,
                 batch_size=100,
                 random_state=1234):
        self.X = X
        self.teacher = teacher
        self.random_state = random_state
        self.batch_size = batch_size
        self.size = len(X)
        self._cur_minibatch = 0

    def generator(self):
        while True:
            for location in iterate_minibatches(self.size,
                                                self.batch_size):
                X_loc = self.X[location]
                y_loc = self.teacher.predict_proba(X_loc)
                yield X_loc, y_loc


def load_from_provider_and_teacher(provider, teacher):
    import numpy as np
    if isinstance(provider, np.ndarray):
        X = provider
        return MiniBatchGenerator(X, teacher,
                                  batch_size=100, random_state=1234)
    else:
        return provider_from_module(provider).provide()
