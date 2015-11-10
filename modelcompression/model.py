import pickle
from urllib.parse import urlparse
from utils import provider_from_module

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def builder(provider):
    model = provider
    return model


def load_from_provider(provider):

    try:
        provider = provider_from_module(provider).provide()
        return provider
    except Exception as e:
        try:
            with open(provider, "rb") as fd:
                model = pickle.load(fd)
            return model
        except Exception as e:
            logging.error(repr(e))
            return provider

def save(model, filename):
    with open(filename, "wb") as fd:
        pickle.dump(model, fd)
