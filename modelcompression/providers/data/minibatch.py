from utils import iterate_minibatches


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
