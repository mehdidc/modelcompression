from lasagne import layers, nonlinearities
from lasagnekit.easy import LightweightModel
from lasagnekit.easy import BatchOptimizer

from hp_toolkit.hp import Model, Param
import batch_optimizer


def build_fully_connected_neural_net(input_shape,
                                     nb_hidden_list,
                                     output_shape,
                                     task='classification'):

    assert isinstance(nb_hidden_list, list)
    assert len(nb_hidden_list) > 0
    assert len(input_shape) > 0
    assert len(output_shape) > 0

    l_in = layers.InputLayer([None] + list(input_shape))

    l_hid = l_in
    for nb_hidden in range(nb_hidden_list):
        l_hid = layers.DenseLayer(l_hid, num_units=nb_hidden)
    l_out = layers.DenseLayer(
        l_hid,
        num_units=output_shape,
        nonlinearity=nonlinearities.softmax)

    return l_in, l_out


class CustomBatchOptimizer(BatchOptimizer):

    def iter_update(self, epoch, nb_batches, iter_update_batch):
        status = super(CustomBatchOptimizer, self).iter_update(
            epoch, nb_batches,
            iter_update_batch)
        status.update(self.model.evaluate_and_get_stats())
        return status

models = dict(
    fully=build_fully_connected_neural_net
)


class Predictor(Model):

    params = dict(
        kind=Param(initial="fully", interval=models.keys(), type='choice')
    )
    params.update(batch_optimizer.params)

    def build_model(self, X, num_classes):
        l_in, l_out = models[self.kind](X.shape, num_classes)
        model = easy.LightweightModel([l_in], [l_out])

        batch_optimizer = MyBatchOptimizer(max_nb_epochs=self.max_epochs,
                optimization_procedure=(updates.adam, {"learning_rate": self.learning_rate}),
                                        verbose=1,
                                        whole_dataset_in_device=True,
                                        batch_size=self.batch_size)
        self.batch_optimizer = batch_optimizer

        def loss_function(model, tensors):
            y, = model.get_output(tensors["X"])
            return objectives.categorical_crossentropy(y, tensors["y"]).mean()

        input_variables = OrderedDict(
            X=dict(tensor_type=T.matrix),
            y=dict(tensor_type=T.ivector)
        )

        functions = dict(
            predict=dict(
                get_output=lambda model, X: model.get_output(X)[0],
                params=["X"]
            ),
        )
        return Capsule(input_variables, model, loss_function,
                       functions=functions,
                       batch_optimizer=batch_optimizer)

    def fit(self, X, y, X_valid=None, y_valid=None):
        num_classes = len(set(y))
        self.model = self.build_model(X, num_classes)
        self.model.X_train = X
        self.model.X_valid = X_valid

        self.batch_optimizer.X_train = X
        self.batch_optimizer.y_train = y
        if X_valid is not None:
            self.batch_optimizer.patience_stat = "error_valid"
            self.batch_optimizer.X_valid = X_valid
            self.batch_optimizer.y_valid = y_valid
        self.model.fit(X=X, y=y)



build_student = build_neural_net_student
