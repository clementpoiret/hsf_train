def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2**k for k in range(num_levels)]
