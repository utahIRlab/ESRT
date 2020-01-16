import tensorflow.compat.v1 as tf
from .get_fs_from_words import get_fs_from_words

def get_query_embedding(model, word_idxs, reuse, scope=None):
    """
    Args:
    Return:
    """
    if 'fs' in model.params['net_struct']:
        return get_fs_from_words(model, word_idxs, reuse, scope)

    else:
        raise ValueError(f"The net struct: \'{model.params['net_struct']}\',  \
                        is not predefined for computing the query embedding.")
