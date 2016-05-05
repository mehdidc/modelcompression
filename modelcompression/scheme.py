from sklearn.metrics import mean_squared_error, accuracy_score

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def standard(model, data, student,
             max_epochs=100,
             evaluator=mean_squared_error):

    generator = data.generator()
    for i in range(max_epochs):
        X, y = next(generator)
        for j in range(10):
            if len(X.shape) > 2:
                X = X.reshape((X.shape[0], -1))
            student.partial_fit(X, y)
            pred = student.predict(X)
            score = evaluator(y, pred)

            message = ("{} at iteration {}: {}".format(
                evaluator.__name__,
                i,
                score))
            logging.info(message)

            accuracy = accuracy_score(y.argmax(axis=1),
                                      pred.argmax(axis=1))
            message = "accuracy : {}".format(accuracy)
            logging.info(message)
