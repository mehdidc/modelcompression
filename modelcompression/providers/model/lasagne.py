import theano
from lasagne import layers


def provide_from_layers(layers_dict):
    assert "input" in layers_dict
    assert "output" in layers_dict

    input_layer = layers_dict["input"]
    if "pre_output" in layers_dict:
        output_layer = layers_dict["pre_output"]
    else:
        output_layer = layers_dict["output"]
    predict_proba = theano.function(
        [input_layer.input_Var],
        layers.get_output(output_layer, input_layer.input_var)
    )

    class Dummy(object):
        pass
    clf = Dummy()
    clf.predict_proba = predict_proba
    return clf
