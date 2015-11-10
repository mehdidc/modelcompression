import theano

class Dummy(object):
    pass

def provide_from_layers(layers_dict):
    assert "input" in layers_dict
    assert "output" in layers_dict

    input_layer = layers_dict["input"]
    if "pre_output" in layers_dict:
        output_layer = layers_dict["pre_output"]
    else:
        output_layer = layers_dict["output"]
    predict_proba = theano.function(
        [X_tensor],
        layers.get_output(output_layer)
    )
    clf = Dummy()
    clf.predict_proba = predict_proba
    return clf
