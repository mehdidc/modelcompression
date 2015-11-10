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
    try:
        scheme.standard(model, data, student, max_epochs=max_epochs)
    except KeyboardInterrupt:
        logging.info("Keyboard interruption, saving the model")
    except Exception as e:
        logging.error(repr(e), exc_info=True)
        return
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
    model = MLPClassifier(verbose=1, hidden_layer_sizes=[100])
    model.fit(X, y)

    compress(model_provider=model,
             data_provider=X,
             student_builder=MLPRegressor())


@task
def googlenet():
    from sklearn.neural_network import MLPRegressor
    compress(model_provider="modelcompression.providers.model.googlenet",
             data_provider="modelcompression.providers.data.image",
             student_builder=MLPRegressor(),
             max_epochs=5000)


@task
def vgg():
    from sklearn.neural_network import MLPRegressor
    compress(model_provider="modelcompression.providers.model.vgg",
             data_provider="modelcompression.providers.data.image",
             student_builder=MLPRegressor(),
             max_epochs=5000)
