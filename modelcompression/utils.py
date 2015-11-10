def iterate_minibatches(nb_inputs, batchsize):
    for start_idx in range(0, nb_inputs - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield excerpt


def provider_from_module(name):
    import importlib
    provider = importlib.import_module(name)
    return provider
