from skimage.io import imread
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
import glob


class Generator(object):

    def __init__(self, teacher, path_generator):
        self.path_generator = path_generator
        self.teacher = teacher

    def generator(self):
        for filenames in self.path_generator:
            X = []
            for filename in filenames:
                logging.info("Loading {}".format(filename))
                try:
                    x = imread(filename)
                except IOError as e:
                    msg = ("Exception when loading the filename"
                           ": {}, skipping".format(repr(e)))
                    logging.warning(msg)
                    continue
                if len(x.shape) != 3 and X.shape[2] != 3:
                    logging.warning("Incompatible shape : {}, skipping".format(x.shape))
                    continue
                logging.debug("shape : {}".format(x.shape))
                X.append(x)
            logging.info("Geting teacher predictions")
            y = self.teacher.predict_proba(X)
            if hasattr(self.teacher, "preprocess"):
                X = self.teacher.preprocess(X)
            yield X, y


class Dummy(object):
    pass


def provide_images_from_folder(teacher, path_pattern, batch_size=10):
    filenames = glob.glob(path_pattern)
    path_generator = retrieve(iter(filenames), batch_size=batch_size)
    return Generator(teacher, path_generator)


def retrieve(generator, batch_size):
    while True:
        d = []
        for i in range(batch_size):
            try:
                new_item = next(generator)
            except StopIteration:
                yield d
                return
            else:
                d.append(new_item)
        yield d


def provide(teacher):
    return provide_images_from_folder(teacher,"/home/mcherti/work/data/imagenet/images/*.jpg",
                                      batch_size=10)
