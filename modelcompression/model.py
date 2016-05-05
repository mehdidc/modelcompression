import pickle
from utils import provider_from_module

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def builder(provider):
    model = provider
    return model


def load_from_provider(provider):
    logging.info("Start Loading model from provider")

    try:
        logging.info("Loading model from a module")
        provider = provider_from_module(provider).provide()
    except Exception as e:
        logging.warning("It is not a module, got an exception : (}".format(repr(e)))
    else:
        return provider

    logging.info("Checking if the provider has partial_fit")
    if hasattr(provider, "partial_fit"):
        logging.info("Found partial_fit in the provider, use it")
        return provider

    logging.info("Try to open it see if it can be pickled")
    try:
        with open(provider, "rb") as fd:
            model = pickle.load(fd)
    except Exception as e:
        logging.error("Could not open it : {}".format(repr(e)), exc_info=True)
    else:
        return model

    raise Exception("Could not load the model from provider, all attemps failed")


def save(model, filename):
    with open(filename, "wb") as fd:
        pickle.dump(model, fd, protocol=pickle.HIGHEST_PROTOCOL)
