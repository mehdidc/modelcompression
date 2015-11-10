from caffezoo.googlenet import GoogleNet
import os

cache_folder = "{}/modelcompression_cache".format(os.getenv("HOME"))


class Dummy(object):
    pass


def provide():
    model_filename = os.path.join(cache_folder, "blvc_googlenet.pkl")
    model = GoogleNet(model_filename=model_filename,
                      layer_names=["loss3/classifier"])

    clf = Dummy()

    def predict_proba(X):
        return model.transform(X)[0]
    clf.predict_proba = predict_proba
    return clf
