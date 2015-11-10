from utils import provider_from_module
import numpy as np

from providers.data.minibatch import MiniBatchGenerator

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_from_provider_and_teacher(provider, teacher):
    logging.info("Start loading data from provider")

    if isinstance(provider, np.ndarray):
        logging.info("It is an np.ndarray, use it")
        X = provider
        return MiniBatchGenerator(X, teacher,
                                  batch_size=100, random_state=1234)

    logging.info("Start Loading provider from a module")

    try:
        provider = provider_from_module(provider)
        data = provider.provide(teacher)
    except Exception as e:
        logging.error("Could not open it : {}".format(repr(e)), exc_info=True)
    else:
        return data

    raise Exception("Could not load the model from provider, all attemps failed")
