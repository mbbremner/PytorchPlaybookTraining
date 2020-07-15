import torch
import torchvision
from torch.utils import data

def select_model(tag, *args, **kwargs):
    return select_module(torchvision.models, tag)(*args, **kwargs)


# def select_optim(tag, model,  *args, **kwargs):
#     return select_module(torch.optim, tag)(
#         model.parameters(), *args, **kwargs)

def select_optim(tag, *args, **kwargs):
    return select_module(torch.optim, tag)(*args, **kwargs)


def select_dataset(tag, *args, **kwargs):
    return select_module(torchvision.datasets, tag)(*args, **kwargs)


def select_loader(tag, *args, **kwargs):
    # return data.DataLoader(dataset, *args, **kwargs)
    return select_module(data, tag)(*args, **kwargs)


def select_scheduler(tag, *args, **kwargs):
    return select_module(torch.optim.lr_scheduler, tag)(*args, **kwargs)


def select_module(module, attr_):
    """ Select a function (attr_) from a module obj (module)"""
    if not hasattr(module, attr_):
        raise TypeError('Bad attr. key: %s, not in the %s' % (attr_, str(module)))
    return getattr(module, attr_)


def init_model_features(model, init_func, *args, **kwargs):
    # 2. Select init func & initialize model params
    for param in model.parameters():
        if len(param.shape) > 1:
            init_func(param, *args, **kwargs)