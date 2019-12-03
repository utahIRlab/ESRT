import tensorflow as tf
from tensorflow.python.ops import variable_scope

from .model_utils import get_query_embedding

def get_product_scores(model, user_idxs, query_word_idx,  product_idxs = None, similarity_func = None,scope = None):
    """compute the score of 'user-query_word' vectors over certain products. if product_idxs=None, we compute the score over all products in
    the dataset.
    Example: user-query-word idxs: [(1,4),
                                    (2,5)]
             product_idxs: [[3],[2],[4]]
            We will get a Shape: (2,3) score matrix. the value at (i, j) means the score of the i-th user-query-word
            over j-th product.

    Args:
        user_idxs: (tf.float with Shape: [None, 1])
        query_word_idx: (tf.float with Shpae [None, 1])
        product_idxs: (tf.float with Shape [None', 1]) if it is none, the product idxs contains
                      all product indexs in our dataset.
        similarity_func: (python function)
        scope: (str) the name scope for all variables in the function

    Return:
        score matrix: (tf.float with Shape: [None, None'])
    """
    with variable_scope.variable_scope(scope or "embedding_graph"):
        # get user embedding [None, embed_size]
        user_vec = tf.nn.embedding_lookup(model.user_emb, user_idxs)
        # get query embedding [None, embed_size]
        query_vec, query_embs = get_query_embedding(model, query_word_idx, True)

        # get candidate product embedding [None, embed_size]
        product_vec = None
        product_bias = None
        if product_idxs != None:
            product_vec = tf.nn.embedding_lookup(model.product_emb, product_idxs)
            product_bias = tf.nn.embedding_lookup(model.product_bias, product_idxs)
        else:
            product_vec = model.product_emb
            product_bias = model.product_bias

        print('Similarity Function : ' + similarity_func)


        if similarity_func == 'product':
            return tf.matmul((1.0 - model.Wu) * user_vec + model.Wu * query_vec, product_vec, transpose_b=True)
        elif similarity_func == 'bias_product':
            return tf.matmul((1.0 - model.Wu) * user_vec + model.Wu * query_vec, product_vec, transpose_b=True) + product_bias
        else:
            user_vec = user_vec / tf.sqrt(tf.reduce_sum(tf.square(user_vec), 1, keep_dims=True))
            query_vec = query_vec / tf.sqrt(tf.reduce_sum(tf.square(query_vec), 1, keep_dims=True))
            product_vec = product_vec / tf.sqrt(tf.reduce_sum(tf.square(product_vec), 1, keep_dims=True))
            return tf.matmul((1.0 - model.Wu) * user_vec + model.Wu * query_vec, product_vec, transpose_b=True)
