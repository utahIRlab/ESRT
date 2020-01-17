import tensorflow.compat.v1 as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

DATA_TYPE = tf.float32

def get_user_vec(model, query_vec, reuse=False, scope=None):
    """Create user embedding based on the query embedding.

    Args:
        model: the model that contains all the functions and parameters.
        query_vec: the query embedding vector.
        reuse: a bool value that decides we should reuse the parameters in the graph.
        scope: the name of the parameter scope.

    Returns:
        A triple consisting of the user embeddings and the parameters for L2 regularization.
    """
    # read user embeddings
    regularization_terms = []
    print('User struct %s' % model.params['user_struct'])

    # define attention function
    def compute_attention_vec(input_vec, att_vecs, lengths, allow_zero_attention=False):
        att_vecs_shape = tf.shape(att_vecs)
        if allow_zero_attention:
            zero_vec = tf.zeros([att_vecs_shape[0], 1, att_vecs_shape[2]], dtype=DATA_TYPE)
            att_vecs_with_zero = tf.concat([att_vecs, zero_vec], 1)
            att_scores, reg_terms = get_attention_scores(input_vec, att_vecs_with_zero, model.params['num_heads'],
                                                             reuse, model.params['attention_func'])
            # att_vecs = tf.Print(att_vecs, [att_vecs], 'this is att_scores', summarize=25)
            regularization_terms.extend(reg_terms)

            # length mask
            mask = tf.concat([tf.sequence_mask(lengths, maxlen=att_vecs_shape[1], dtype=DATA_TYPE),
                              tf.ones([att_vecs_shape[0], 1], dtype=DATA_TYPE)], 1)
            masked_att_exp = tf.exp(att_scores) * mask
            div = tf.reduce_sum(masked_att_exp, 1, True)
            att_dis = masked_att_exp / tf.where(tf.less(div, 1e-7), div+1, div)
            model.attention_distribution = att_dis
            return tf.reduce_sum(att_vecs_with_zero * tf.expand_dims(att_dis, -1), 1)

        else:
            att_scores, reg_terms = get_attention_scores(input_vec, att_vecs, model.params['num_heads'],
                                                             reuse, model.params['attention_func'])
            # att_vecs = tf.Print(att_vecs, [att_vecs], 'this is att_scores', summarize=25)
            regularization_terms.extend(reg_terms)

            # length mask
            mask = tf.sequence_mask(lengths, maxlen=att_vecs_shape[1], dtype=DATA_TYPE)
            masked_att_exp = tf.exp(att_scores) * mask
            div = tf.reduce_sum(masked_att_exp, 1, True)
            att_dis = masked_att_exp / tf.where(tf.less(div, 1e-7), div + 1, div)
            '''
            att_exp = tf.exp(att_scores)
            att_dis = att_exp/(tf.reduce_sum(att_exp, 1, True) + tf.expand_dims(lengths, -1) - model.params.max_history_length)
            '''
            # att_dis = tf.Print(att_dis, [att_dis], 'this is att_dis', summarize=25)
            model.attention_distribution = att_dis
            return tf.reduce_sum(att_vecs * tf.expand_dims(att_dis, -1), 1)

    # get user embedding from asin history
    user_vec = None
    if model.params['user_struct'] == 'asin_mean':
        history_length = model.input_history_length
        mask = tf.expand_dims(tf.sequence_mask(history_length, maxlen=model.params['max_history_length'], dtype=DATA_TYPE), -1)
        asin_idxs_list = model.history_product_idxs
        asin_vecs, r_terms = get_asin_vec(model, asin_idxs_list, True)
        regularization_terms.extend(r_terms)
        div = tf.to_float(tf.expand_dims(history_length, -1))
        user_vec = tf.reduce_sum(asin_vecs * mask, 1) / tf.where(tf.less(div, 1e-7), div+1, div)
    elif model.params['user_struct'] == 'asin_zero_mean':
        history_length = model.input_history_length
        mask = tf.expand_dims(tf.sequence_mask(history_length, maxlen=model.params['max_history_length'], dtype=DATA_TYPE), -1)
        asin_idxs_list = model.history_product_idxs
        asin_vecs, r_terms = get_asin_vec(model, asin_idxs_list, True)
        regularization_terms.extend(r_terms)
        div = tf.to_float(tf.expand_dims(history_length, -1))
        mean_vec = tf.reduce_sum(asin_vecs * mask, 1) / tf.where(tf.less(div, 1e-7), div+1, div)
        user_vec = compute_attention_vec(query_vec, tf.expand_dims(mean_vec, 1), tf.ones_like(history_length), True)

    elif model.params['user_struct'] == 'asin_attention':
        history_length = model.input_history_length
        asin_idxs_list = model.history_product_idxs
        #model.print_ops.append(tf.print("history_length: ", model.max_history_length))
        #model.print_ops.append(tf.print("asin_idxs_list: ", model.history_product_idxs))

        with variable_scope.variable_scope(scope or 'history_attention', reuse=reuse):
            asin_vecs, r_terms = get_asin_vec(model, asin_idxs_list, True)
            regularization_terms.extend(r_terms)
            user_vec = compute_attention_vec(query_vec, asin_vecs, history_length)

    elif model.params['user_struct'] == 'asin_zero_attention':
        history_length = model.input_history_length
        asin_idxs_list = model.history_product_idxs
        with variable_scope.variable_scope(scope or 'history_attention', reuse=reuse):
            asin_vecs, r_terms = get_asin_vec(model, asin_idxs_list, True)
            regularization_terms.extend(r_terms)
            user_vec = compute_attention_vec(query_vec, asin_vecs, history_length, True)

    elif model.params['user_struct'] == 'query_mean':
        history_length = model.input_history_length
        mask = tf.expand_dims(
            tf.sequence_mask(history_length, maxlen=model.params['max_history_length'], dtype=DATA_TYPE), -1)
        query_idxs_list = model.history_query_idxs
        query_word_idxs = tf.nn.embedding_lookup(model.query_word_list, query_idxs_list)
        query_length = tf.nn.embedding_lookup(model.query_lengths, query_idxs_list)
        # query_word_idxs = tf.Print(query_word_idxs, [query_word_idxs], 'this is query_word_idxs', summarize=5)
        query_embs, r_terms = get_embedding_from_words(model, query_word_idxs,
                                                           query_length,
                                                           model.params['query_embed_func'],
                                                           tf.AUTO_REUSE, scope='query_embed')
        regularization_terms.extend(r_terms)
        div = tf.to_float(tf.expand_dims(history_length, -1))
        user_vec = tf.reduce_sum(query_embs * mask, 1) / tf.where(tf.less(div, 1e-7), div+1, div)

    elif model.params['user_struct'] == 'query_attention':
        history_length = model.input_history_length
        query_idxs_list = model.history_query_idxs
        with variable_scope.variable_scope(scope or 'history_attention', reuse=reuse):
            query_word_idxs = tf.nn.embedding_lookup(model.query_word_list, query_idxs_list)
            query_length = tf.nn.embedding_lookup(model.query_lengths, query_idxs_list)
            # query_word_idxs = tf.Print(query_word_idxs, [query_word_idxs], 'this is query_word_idxs', summarize=5)
            query_embs, r_terms = get_embedding_from_words(model, query_word_idxs,
                                                               query_length,
                                                               model.params['query_embed_func'],
                                                               tf.AUTO_REUSE, scope='query_embed')
            regularization_terms.extend(r_terms)
            user_vec = compute_attention_vec(query_vec, query_embs, history_length, True)

    elif model.params['user_struct'] == 'query_zero_attention':
        history_length = model.input_history_length
        query_idxs_list = model.history_query_idxs
        with variable_scope.variable_scope(scope or 'history_attention', reuse=reuse):
            query_word_idxs = tf.nn.embedding_lookup(model.query_word_list, query_idxs_list)
            query_length = tf.nn.embedding_lookup(model.query_lengths, query_idxs_list)
            # query_word_idxs = tf.Print(query_word_idxs, [query_word_idxs], 'this is query_word_idxs', summarize=5)
            query_embs, r_terms = get_embedding_from_words(model, query_word_idxs,
                                                               query_length,
                                                               model.params['query_embed_func'],
                                                               tf.AUTO_REUSE, scope='query_embed')
            regularization_terms.extend(r_terms)
            user_vec = compute_attention_vec(query_vec, query_embs, history_length)

    elif model.params['user_struct'] == 'asin_query_concat':
        history_length = model.input_history_length
        div = tf.to_float(tf.expand_dims(history_length, -1))
        mask = tf.expand_dims(
            tf.sequence_mask(history_length, maxlen=model.params['max_history_length'], dtype=DATA_TYPE), -1)
        asin_idxs_list = model.history_product_idxs
        query_idxs_list = model.history_query_idxs
        asin_vecs, r_terms = get_asin_vec(model, asin_idxs_list, True)
        regularization_terms.extend(r_terms)

        user_vec = tf.reduce_sum(asin_vecs * mask, 1) / tf.where(tf.less(div, 1e-7), div+1, div)

        query_word_idxs = tf.nn.embedding_lookup(model.query_word_list, query_idxs_list)
        query_length = tf.nn.embedding_lookup(model.query_lengths, query_idxs_list)
        # query_word_idxs = tf.Print(query_word_idxs, [query_word_idxs], 'this is query_word_idxs', summarize=5)
        query_embs, r_terms = get_embedding_from_words(model, query_word_idxs,
                                                           query_length,
                                                           model.params['query_embed_func'],
                                                           tf.AUTO_REUSE, scope='query_embed')
        regularization_terms.extend(r_terms)
        user_vec = tf.concat(
            [user_vec, tf.reduce_sum(query_embs * mask, 1) / tf.where(tf.less(div, 1e-7), div+1, div)],
            1)

    elif model.params['user_struct'] == 'none':
        asin_idxs_list = model.model.history_product_idxs
        asin_vecs, r_terms = get_asin_vec(model, asin_idxs_list, True)
        user_vec = tf.zeros_like(tf.reduce_sum(asin_vecs, 1))

    elif model.params['user_struct'] == 'user_abstract':
        user_vec = tf.nn.embedding_lookup(model.user_emb, model.user_idxs)
        combined_emb = tf.concat([user_vec, query_vec], 1)
        user_vec, r_terms = dnn_abstract(combined_emb, model.params['embed_size'], reuse,
                                                     'user_query_abstract', tf.nn.elu)
        regularization_terms.extend(r_terms)

    elif model.params['user_struct'] == 'user_attention':
        user_vec = tf.nn.embedding_lookup(model.user_emb, model.user_idxs)
        user_vec = tf.expand_dims(user_vec, 1)
        batch_size = tf.shape(user_vec)[0]
        with variable_scope.variable_scope(scope or 'user_attention', reuse=reuse):
            user_vec = compute_attention_vec(query_vec, user_vec, tf.ones([batch_size]), True)
        regularization_terms.append(user_vec)

    else:
        sys.exit('Error! No user embedding method named %s' % model.params['user_struct'])
    # print(user_vec.get_shape())
    #user_vec = tf.Print(user_vec, [user_vec], 'this is user_vec', summarize=50)
    return user_vec, regularization_terms

def get_asin_vec(model, asin_idxs_list, reuse=False):
    # TOOD: have more get vec's method
    regularization_terms = []
    asin_vecs = None

    asin_vecs = tf.nn.embedding_lookup(model.product_emb, asin_idxs_list)
    regularization_terms.append(asin_vecs)

    return asin_vecs, regularization_terms

def get_attention_scores(current_state, attention_vecs, num_heads, reuse=False, attention_func='default'):
    """Compute a single vector output over a list of words with the corresponding embedding function.

    Args:
        current_state: the state vector used to compute attention.
        attention_vecs: a list of vectors to attend.
        num_heads: the number of heads in the attention function.
        reuse: a bool value that decides we should reuse the parameters in the graph.
        attention_func: the name of the attention function.

    Returns:
        A triple consisting of the attention scores and the parameters for L2 regularization.

    """
    state_size = current_state.get_shape()[1]  # the dimension size of state vector
    attn_size = attention_vecs.get_shape()[2]  # the dimension size of each output vector
    print("state_size: ", state_size, "attention size: ", attn_size)
    att_score_list = []
    regularization_terms = []
    if attention_func == 'dot':
        print('Attention function: %s' % attention_func)
        s = tf.reduce_sum(attention_vecs * tf.reshape(current_state, [-1, 1, attn_size]), axis=2)
        head_weight = variable_scope.get_variable("head_weight", [1], dtype=DATA_TYPE)
        att_score_list.append(s * head_weight)
        regularization_terms.append(head_weight)
    else:
        for a in range(num_heads):
            with variable_scope.variable_scope('Att_%d' % a, reuse=reuse, dtype=DATA_TYPE):
                W = variable_scope.get_variable("linear_W_%d" % a, [state_size, attn_size])
                b = variable_scope.get_variable("linear_b_%d" % a, [attn_size])
                y = tf.nn.bias_add(tf.matmul(current_state, W), b)  # W*s+b
                y = tf.reshape(y, [-1, 1, attn_size])
                s = math_ops.reduce_sum(attention_vecs * tf.tanh(y), axis=2)  # o * tanh(W*s+b)
                regularization_terms.append(W)
                head_weight = variable_scope.get_variable("head_weight_%d" % a, [1])
                att_score_list.append(s * head_weight)
                regularization_terms.append(head_weight)

    att_scores = tf.reduce_sum(att_score_list, axis=0)
    att_scores = att_scores - tf.reduce_max(att_scores, 1, True)
    return att_scores, regularization_terms
