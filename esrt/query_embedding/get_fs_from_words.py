

import tensorflow.compat.v1 as tf
from tensorflow.python.ops import variable_scope

def get_fs_from_words(model, word_idxs, word_emb, reuse, scope=None):
    """
    Get the embedding from word_idxs.

    Args:
        model: esrt.engine.Model.
        word_idxs: Tensor with shape of [batch_size, query_max_length]
                    with type of int32.
        reuse: bool.
        scope: str.

    Return:
        f_s: Tensor with shape of [batch_size, embed_size] with type of float32
        [f_W, word_vecs]: List of two Tensor:
                          f_W: Tensor with shape of [embed_size, embed_size] with type of float32.
                          word_vecs: Tensor with shape of [batch_size, query_max_length, embed_size]
                                     with type of float32.
    """
    with variable_scope.variable_scope(scope or 'f_s_abstraction',
     reuse=reuse):
    # get mean word vectors
        word_vecs = tf.nn.embedding_lookup(word_emb, word_idxs)
        mean_word_vec = tf.reduce_mean(word_vecs, 1)
        # get f(s)
        f_W = variable_scope.get_variable("f_W", [model.embed_size, model.embed_size])
        f_b = variable_scope.get_variable("f_b", [model.embed_size])
        f_s = tf.tanh(tf.nn.bias_add(tf.matmul(mean_word_vec, f_W), f_b))
        return f_s, [f_W, word_vecs]
