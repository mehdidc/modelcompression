from invoke import task

import model as model_funcs
import data as data_funcs
import scheme

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task
def compress(model_provider,
             data_provider,
             student_builder,
             max_epochs=10,
             out_filename="compressed.pkl"):

    logger.info("Loading teacher model")
    model = model_funcs.load_from_provider(model_provider)
    logger.info("Loading data")
    data = data_funcs.load_from_provider_and_teacher(data_provider,
                                                     teacher=model)
    logger.info("Building student model")
    student = model_funcs.builder(student_builder)
    logger.info("Start compression...")
    scheme.standard(model, data, student, max_epochs=max_epochs)
    logger.info("Saving the compressed model")

    student.meta = dict(
        model_provider=repr(model_provider),
        student_builder=repr(student_builder),
        data_provider=repr(data_provider)
    )
    model_funcs.save(student, out_filename)


@task
def test():
    from sklearn.datasets import load_digits
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    data = load_digits()
    X = data["data"]
    y = data["target"]
    model = MLPClassifier(verbose=1)
    model.fit(X, y)

    compress(model_provider=model,
             data_provider=X,
             student_builder=MLPRegressor())
