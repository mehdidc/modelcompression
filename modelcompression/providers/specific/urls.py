import numpy as np
import pandas as pd

import urllib
from skimage.io import imread


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Generator(object):

    def __init__(self, teacher, url_generator):
        self.url_generator = url_generator
        self.teacher = teacher

    def generate(self):
        urls = self.url_generator()
        X = []
        for url in urls:
            try:
                logging.info("Downloading {}".format(url))
                urllib.urlretrieve(url, "tmpfile")
                x = imread("tmpfile")
            except Exception as e:
                logging.error(repr(e))
            else:
                X.append(x[None, :, :, :])
        X = np.concatenate(X, axis=0).astype(np.float32)
        y = self.teacher.predict_proba(X)
        yield X, y

def provide_images_from_urls(teacher, filename, batch_size=10):
    gen = pd.read_csv(filename, chunksize=batch_size)

    def url_generator():
        return gen.get_chunk().iloc[:, 1]

    return Generator(teacher, url_generator)
