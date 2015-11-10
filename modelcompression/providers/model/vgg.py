from caffezoo.vgg import VGG
import os

cache_folder = "{}/modelcompression_cache".format(os.getenv("HOME"))


class Dummy(object):
    pass


def provide():
    model_filename = os.path.join(cache_folder, "vgg19.pkl")
    model = VGG(model_filename=model_filename,
                layer_names=["fc8"])
    model._load()
    clf = Dummy()

    def predict_proba(X):
        pre_softmax_layer = model.transform(X)
        return pre_softmax_layer

    def preprocess(X):
        return model.preprocess(X)
    clf.predict_proba = predict_proba
    clf.preprocess = preprocess
    return clf
