from collections import OrderedDict
from lasagne import layers, nonlinearities, updates
from lasagnekit.easy import LightweightModel
from lasagnekit.easy import BatchOptimizer
from lasagnekit.generative.capsule import Capsule

import theano.tensor as T

from hp_toolkit.hp import Model, Param
import batch_optimizer


def build_fully_connected_neural_net(input_shape,
                                     nb_hidden_list,
                                     output_size):
    assert isinstance(nb_hidden_list, list)
    assert len(nb_hidden_list) > 0
    assert len(input_shape) > 0

    l_in = layers.InputLayer([None] + list(input_shape[1:]))

    l_hid = l_in
    for nb_hidden in nb_hidden_list:
        l_hid = layers.DenseLayer(l_hid, num_units=nb_hidden)
    l_out = layers.DenseLayer(
        l_hid,
        num_units=output_size)
    return l_in, l_out


class CustomBatchOptimizer(BatchOptimizer):

    def iter_update(self, epoch, nb_batches, iter_update_batch):
        status = super(CustomBatchOptimizer, self).iter_update(
            epoch, nb_batches,
            iter_update_batch)
        return status

models = dict(
    fully=build_fully_connected_neural_net
)


def get_output(model, X) :
    return model.get_output(X)[0]


class Regressor(Model):

    params = dict(
        kind=Param(initial="fully", interval=models.keys(), type='choice'),
        nb_units=Param(initial=100, interval=[100, 200], type='int'),
        nb_layers=Param(initial=1, interval=[1, 2, 3, 4], type='choice')
    )
    params.update(batch_optimizer.params)

    def build_model(self, X, num_outputs):
        hidden_sizes = [self.nb_units] * self.nb_layers
        l_in, l_out = models[self.kind](X.shape, hidden_sizes, num_outputs)
        model = LightweightModel([l_in], [l_out])

        batch_optimizer = CustomBatchOptimizer(max_nb_epochs=self.max_epochs,
                optimization_procedure=(updates.adam, {"learning_rate": self.learning_rate}),
                                        verbose=0,
                                        whole_dataset_in_device=False,
                                        batch_size=self.batch_size)
        self.batch_optimizer = batch_optimizer

        def loss_function(model, tensors):
            y, = model.get_output(tensors["X"])
            return (0.5 * (y - tensors["y"])**2).sum(axis=1).mean()

        input_variables = OrderedDict(
            X=dict(tensor_type=T.matrix),
            y=dict(tensor_type=T.matrix)
        )

        functions = dict(
            predict=dict(
                get_output=get_output,
                params=["X"]
            ),
        )
        return Capsule(input_variables, model, loss_function,
                       functions=functions,
                       batch_optimizer=batch_optimizer)

    def fit(self, X, y, X_valid=None, y_valid=None):
        if not hasattr(self, "model"):
            self.model = self.build_model(X, num_outputs=y.shape[1])
        self.model.X_train = X
        self.model.X_valid = X_valid

        self.batch_optimizer.X_train = X
        self.batch_optimizer.y_train = y
        if X_valid is not None:
            self.batch_optimizer.patience_stat = "error_valid"
            self.batch_optimizer.X_valid = X_valid
            self.batch_optimizer.y_valid = y_valid
        self.model.fit(X=X, y=y)

    def partial_fit(self, X, y):
        self.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
