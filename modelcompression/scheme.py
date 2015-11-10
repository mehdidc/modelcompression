from sklearn.metrics import mean_squared_error

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def standard(model, data, student,
             max_epochs=100,
             evaluator=mean_squared_error):


    generator = data.generator()
    for i in range(max_epochs):
        X, y = next(generator)
        student.partial_fit(X, y)
        score = evaluator(y, student.predict(X))

        message = ("{} at iteration {}: {}".format(
            evaluator.__name__,
            i,
            score))
        logging.info(message)
