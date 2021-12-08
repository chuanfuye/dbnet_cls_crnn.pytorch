from ._360cc import _360CC
from ._icdar2015 import _icdar2015
from ._own import _OWN


def get_dataset(config):

    if config.DATASET.DATASET == "360CC":
        return _360CC
    elif config.DATASET.DATASET == "icdar2015":
        return _icdar2015
    else:
        raise NotImplemented()
