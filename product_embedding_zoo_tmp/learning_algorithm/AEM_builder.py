from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange# pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange# pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf

DATA_TYPE = tf.float32

def get_product_scores(model, user_idxs, query_word_idx, product_idxs = None, scope = None):
    with variable_scope.variable_scope(scope or "embedding_graph"):
        # get user embedding [None, embed_size]
        #user_vec = tf.nn.embedding_lookup(model.user_emb, user_idxs)
        # get query embedding [None, embed_size]
        query_vec, query_embs = get_query_embedding(model, query_word_idx, True)
        user_vec, _ = get_user_vec(model, query_vec, True)
        
        # get candidate product embedding [None, embed_size]
        product_vec = None
        product_bias = None
        if product_idxs != None:
            product_vec = tf.nn.embedding_lookup(model.product_emb, product_idxs)
            product_bias = tf.nn.embedding_lookup(model.product_bias, product_idxs)
        else:
            product_vec = model.product_emb
            product_bias = model.product_bias

        print('Similarity Function : ' + model.similarity_func)


        if model.similarity_func == 'product':
            return tf.matmul((1.0 - model.Wu) * user_vec + model.Wu * query_vec, product_vec, transpose_b=True)
        elif model.similarity_func == 'bias_product':
            return tf.matmul((1.0 - model.Wu) * user_vec + model.Wu * query_vec, product_vec, transpose_b=True) + product_bias
        else:
            user_vec = user_vec / tf.sqrt(tf.reduce_sum(tf.square(user_vec), 1, keep_dims=True))
            query_vec = query_vec / tf.sqrt(tf.reduce_sum(tf.square(query_vec), 1, keep_dims=True))
            product_vec = product_vec / tf.sqrt(tf.reduce_sum(tf.square(product_vec), 1, keep_dims=True))
            return tf.matmul((1.0 - model.Wu) * user_vec + model.Wu * query_vec, product_vec, transpose_b=True)

def get_fs_from_words(model, word_idxs, reuse, scope=None):
    with variable_scope.variable_scope(scope or 'f_s_abstraction',
                                         reuse=reuse):
        # get mean word vectors
        word_vecs = tf.nn.embedding_lookup(model.word_emb, word_idxs)
        mean_word_vec = tf.reduce_mean(word_vecs, 1)
        # get f(s)
        f_W = variable_scope.get_variable("f_W", [model.embed_size, model.embed_size])
        f_b = variable_scope.get_variable("f_b", [model.embed_size])
        f_s = tf.tanh(tf.nn.bias_add(tf.matmul(mean_word_vec, f_W), f_b))
        return f_s, [f_W, word_vecs]

def get_addition_from_words(model, word_idxs, reuse, scope=None):
    with variable_scope.variable_scope(scope or 'addition_abstraction',
                                         reuse=reuse):
        # get mean word vectors
        word_vecs = tf.nn.embedding_lookup(model.word_emb, word_idxs)
        mean_word_vec = tf.reduce_mean(word_vecs, 1)
        return mean_word_vec, [word_vecs]

def get_RNN_from_words(model, word_idxs, reuse, scope=None):
    with variable_scope.variable_scope(scope or 'RNN_abstraction',
                                         reuse=reuse):
        # get mean word vectors
        word_vecs = tf.nn.embedding_lookup(model.word_emb, word_idxs)
        cell = tf.contrib.rnn.GRUCell(model.embed_size)
        encoder_outputs, encoder_state = tf.nn.static_rnn(cell, tf.unstack(word_vecs, axis=1), dtype=dtypes.float32)
        return encoder_state, [word_vecs]

def get_attention_from_words(model, word_idxs, reuse, scope=None):
    with variable_scope.variable_scope(scope or 'attention_abstraction',
                                         reuse=reuse,
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)):
        # get mean word vectors
        word_vecs = tf.nn.embedding_lookup(model.word_emb, word_idxs) # [batch,query_max_length,embed_size]
        #print(word_vecs.get_shape())
        # build mask
        mask = tf.maximum(tf.cast(word_idxs, tf.float32) + 1.0, 1.0) # [batch,query_max_length]
        # softmax weight
        #print(word_idxs.get_shape())
        gate_W = variable_scope.get_variable("gate_W", [model.embed_size])
        #print(tf.reduce_sum(word_vecs * gate_W,2).get_shape())
        word_weight = tf.exp(tf.reduce_sum(word_vecs * gate_W,2)) * mask
        word_weight = word_weight / tf.reduce_sum(word_weight,1,keep_dims=True)
        # weigted sum
        att_word_vec = tf.reduce_sum(word_vecs * tf.expand_dims(word_weight,2),1)
        return att_word_vec, [word_vecs]


def get_query_embedding(model, word_idxs, reuse, scope = None):
    if 'mean' in model.net_struct: # mean vector
        print('Query model: mean')
        return get_addition_from_words(model, word_idxs, reuse, scope)
    elif 'fs' in model.net_struct: # LSE f(s)
        print('Query model: LSE f(s)')
        return get_fs_from_words(model, word_idxs, reuse, scope)
    elif 'RNN' in model.net_struct: # RNN
        print('Query model: RNN')
        return get_RNN_from_words(model, word_idxs, reuse, scope)
    else:
        print('Query model: Attention')
        return get_attention_from_words(model, word_idxs, reuse, scope)


def build_embedding_graph_and_loss(model, scope = None):
    with variable_scope.variable_scope(scope or "embedding_graph"):
        # Word embeddings.
        init_width = 0.5 / model.embed_size
        model.word_emb = tf.Variable( tf.random_uniform(
                            [model.vocab_size+1, model.embed_size], -init_width, init_width),
                            name="word_emb")
        #model.word_emb = tf.concat(axis=0,values=[model.word_emb,model.PAD_embed])
        model.word_bias = tf.Variable(tf.zeros([model.vocab_size+1]), name="word_b")

        # user/product embeddings.
        model.user_emb =    tf.Variable( tf.zeros([model.user_size, model.embed_size]),
                            name="user_emb")
        model.user_bias =    tf.Variable( tf.zeros([model.user_size]), name="user_b")
        model.product_emb =    tf.Variable( tf.zeros([model.product_size+1, model.embed_size]),
                            name="product_emb")
        model.product_bias =    tf.Variable( tf.zeros([model.product_size+1]), name="product_b")
        # Review embeddings.
        if model.need_review:
            model.review_emb = tf.Variable( tf.zeros([model.review_size, model.embed_size]),
                                name="review_emb")
            model.review_bias = tf.Variable(tf.zeros([model.review_size]), name="review_b")

        if model.need_context:
            # Context embeddings.
            model.context_emb = tf.Variable( tf.zeros([model.vocab_size, model.embed_size]),
                                name="context_emb")
            model.context_bias = tf.Variable(tf.zeros([model.vocab_size]), name="context_b")
            return UQP_nce_loss(model, model.user_idxs, model.query_word_idxs, model.product_idxs, model.review_idxs,
                                        model.word_idxs, model.context_word_idxs)

        loss = None
        regularization_terms = []
        # product predict words loss
        pr_loss, pr_embs = single_nce_loss(model, model.product_idxs, model.product_emb, model.word_idxs, model.word_emb,
                                            model.word_bias, model.vocab_size, model.vocab_distribute)
        loss = pr_loss
        regularization_terms += pr_embs
        #model.print_ops.append(tf.print("product_loss: ", pr_loss, output_stream=sys.stdout))

        # get query_vec
        query_vec, qw_embs = get_query_embedding(model, model.query_word_idxs, None)
        regularization_terms += qw_embs

        # get user_vec by looking history product list
        user_vec, r_terms = get_user_vec(model, query_vec)
        regularization_terms += r_terms
        regularization_terms.append(user_vec)

        # compute the pair search loss
        combined_vec = user_vec * (1-model.Wu) + query_vec * model.Wu
        uqr_loss, uqr_embs = single_nce_loss_with_vec(model, combined_vec, model.product_idxs, model.product_emb,
                            model.product_bias, model.product_size, model.product_distribute)
        loss += uqr_loss
        regularization_terms += uqr_embs

        # L2 regularization
        if model.L2_lambda > 0:
            l2_loss = tf.nn.l2_loss(regularization_terms[0])
            for i in xrange(1,len(regularization_terms)):
                l2_loss += tf.nn.l2_loss(regularization_terms[i])
            loss += model.L2_lambda * l2_loss

        batch_size = array_ops.shape(model.word_idxs)[0]  #get batch_size
        #model.print_ops.append(tf.print("total_loss: ", loss, output_stream=sys.stdout))
        #model.print_ops.append(tf.print("batch size: ", batch_size, output_stream=sys.stdout))
        return loss / math_ops.cast(batch_size, dtypes.float32)

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
    print('User struct %s' % model.hparams.user_struct)

    # define attention function
    def compute_attention_vec(input_vec, att_vecs, lengths, allow_zero_attention=False):
        att_vecs_shape = tf.shape(att_vecs)
        if allow_zero_attention:
            zero_vec = tf.zeros([att_vecs_shape[0], 1, att_vecs_shape[2]], dtype=DATA_TYPE)
            att_vecs_with_zero = tf.concat([att_vecs, zero_vec], 1)
            att_scores, reg_terms = get_attention_scores(input_vec, att_vecs_with_zero, model.hparams.num_heads,
                                                             reuse, model.hparams.attention_func)
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
            att_scores, reg_terms = get_attention_scores(input_vec, att_vecs, model.hparams.num_heads,
                                                             reuse, model.hparams.attention_func)
            # att_vecs = tf.Print(att_vecs, [att_vecs], 'this is att_scores', summarize=25)
            regularization_terms.extend(reg_terms)

            # length mask
            mask = tf.sequence_mask(lengths, maxlen=att_vecs_shape[1], dtype=DATA_TYPE)
            masked_att_exp = tf.exp(att_scores) * mask
            div = tf.reduce_sum(masked_att_exp, 1, True)
            att_dis = masked_att_exp / tf.where(tf.less(div, 1e-7), div + 1, div)
            '''
            att_exp = tf.exp(att_scores)
            att_dis = att_exp/(tf.reduce_sum(att_exp, 1, True) + tf.expand_dims(lengths, -1) - model.hparams.max_history_length)
            '''
            # att_dis = tf.Print(att_dis, [att_dis], 'this is att_dis', summarize=25)
            model.attention_distribution = att_dis
            return tf.reduce_sum(att_vecs * tf.expand_dims(att_dis, -1), 1)

    # get user embedding from asin history
    user_vec = None
    if model.hparams.user_struct == 'asin_mean':
        history_length = model.input_history_length
        mask = tf.expand_dims(tf.sequence_mask(history_length, maxlen=model.hparams.max_history_length, dtype=DATA_TYPE), -1)
        asin_idxs_list = model.history_asin_idxs
        asin_vecs, r_terms = get_asin_vec(model, asin_idxs_list, True)
        regularization_terms.extend(r_terms)
        div = tf.to_float(tf.expand_dims(history_length, -1))
        user_vec = tf.reduce_sum(asin_vecs * mask, 1) / tf.where(tf.less(div, 1e-7), div+1, div)
    elif model.hparams.user_struct == 'asin_zero_mean':
        history_length = model.input_history_length
        mask = tf.expand_dims(tf.sequence_mask(history_length, maxlen=model.hparams.max_history_length, dtype=DATA_TYPE), -1)
        asin_idxs_list = model.history_asin_idxs
        asin_vecs, r_terms = get_asin_vec(model, asin_idxs_list, True)
        regularization_terms.extend(r_terms)
        div = tf.to_float(tf.expand_dims(history_length, -1))
        mean_vec = tf.reduce_sum(asin_vecs * mask, 1) / tf.where(tf.less(div, 1e-7), div+1, div)
        user_vec = compute_attention_vec(query_vec, tf.expand_dims(mean_vec, 1), tf.ones_like(history_length), True)

    elif model.hparams.user_struct == 'asin_attention':
        history_length = model.max_history_length
        asin_idxs_list = model.history_product_idxs
        #model.print_ops.append(tf.print("history_length: ", model.max_history_length))
        #model.print_ops.append(tf.print("asin_idxs_list: ", model.history_product_idxs))

        with variable_scope.variable_scope(scope or 'history_attention', reuse=reuse):
            asin_vecs, r_terms = get_asin_vec(model, asin_idxs_list, True)
            regularization_terms.extend(r_terms)
            user_vec = compute_attention_vec(query_vec, asin_vecs, history_length)

    elif model.hparams.user_struct == 'asin_zero_attention':
        history_length = model.input_history_length
        asin_idxs_list = model.history_asin_idxs
        with variable_scope.variable_scope(scope or 'history_attention', reuse=reuse):
            asin_vecs, r_terms = get_asin_vec(model, asin_idxs_list, True)
            regularization_terms.extend(r_terms)
            user_vec = compute_attention_vec(query_vec, asin_vecs, history_length, True)

    elif model.hparams.user_struct == 'query_mean':
        history_length = model.input_history_length
        mask = tf.expand_dims(
            tf.sequence_mask(history_length, maxlen=model.hparams.max_history_length, dtype=DATA_TYPE), -1)
        query_idxs_list = model.history_query_idxs
        query_word_idxs = tf.nn.embedding_lookup(model.query_word_list, query_idxs_list)
        query_length = tf.nn.embedding_lookup(model.query_lengths, query_idxs_list)
        # query_word_idxs = tf.Print(query_word_idxs, [query_word_idxs], 'this is query_word_idxs', summarize=5)
        query_embs, r_terms = get_embedding_from_words(model, query_word_idxs,
                                                           query_length,
                                                           model.hparams.query_embed_func,
                                                           tf.AUTO_REUSE, scope='query_embed')
        regularization_terms.extend(r_terms)
        div = tf.to_float(tf.expand_dims(history_length, -1))
        user_vec = tf.reduce_sum(query_embs * mask, 1) / tf.where(tf.less(div, 1e-7), div+1, div)

    elif model.hparams.user_struct == 'query_attention':
        history_length = model.input_history_length
        query_idxs_list = model.history_query_idxs
        with variable_scope.variable_scope(scope or 'history_attention', reuse=reuse):
            query_word_idxs = tf.nn.embedding_lookup(model.query_word_list, query_idxs_list)
            query_length = tf.nn.embedding_lookup(model.query_lengths, query_idxs_list)
            # query_word_idxs = tf.Print(query_word_idxs, [query_word_idxs], 'this is query_word_idxs', summarize=5)
            query_embs, r_terms = get_embedding_from_words(model, query_word_idxs,
                                                               query_length,
                                                               model.hparams.query_embed_func,
                                                               tf.AUTO_REUSE, scope='query_embed')
            regularization_terms.extend(r_terms)
            user_vec = compute_attention_vec(query_vec, query_embs, history_length, True)

    elif model.hparams.user_struct == 'query_zero_attention':
        history_length = model.input_history_length
        query_idxs_list = model.history_query_idxs
        with variable_scope.variable_scope(scope or 'history_attention', reuse=reuse):
            query_word_idxs = tf.nn.embedding_lookup(model.query_word_list, query_idxs_list)
            query_length = tf.nn.embedding_lookup(model.query_lengths, query_idxs_list)
            # query_word_idxs = tf.Print(query_word_idxs, [query_word_idxs], 'this is query_word_idxs', summarize=5)
            query_embs, r_terms = get_embedding_from_words(model, query_word_idxs,
                                                               query_length,
                                                               model.hparams.query_embed_func,
                                                               tf.AUTO_REUSE, scope='query_embed')
            regularization_terms.extend(r_terms)
            user_vec = compute_attention_vec(query_vec, query_embs, history_length)

    elif model.hparams.user_struct == 'asin_query_concat':
        history_length = model.input_history_length
        div = tf.to_float(tf.expand_dims(history_length, -1))
        mask = tf.expand_dims(
            tf.sequence_mask(history_length, maxlen=model.hparams.max_history_length, dtype=DATA_TYPE), -1)
        asin_idxs_list = model.history_asin_idxs
        query_idxs_list = model.history_query_idxs
        asin_vecs, r_terms = get_asin_vec(model, asin_idxs_list, True)
        regularization_terms.extend(r_terms)

        user_vec = tf.reduce_sum(asin_vecs * mask, 1) / tf.where(tf.less(div, 1e-7), div+1, div)

        query_word_idxs = tf.nn.embedding_lookup(model.query_word_list, query_idxs_list)
        query_length = tf.nn.embedding_lookup(model.query_lengths, query_idxs_list)
        # query_word_idxs = tf.Print(query_word_idxs, [query_word_idxs], 'this is query_word_idxs', summarize=5)
        query_embs, r_terms = get_embedding_from_words(model, query_word_idxs,
                                                           query_length,
                                                           model.hparams.query_embed_func,
                                                           tf.AUTO_REUSE, scope='query_embed')
        regularization_terms.extend(r_terms)
        user_vec = tf.concat(
            [user_vec, tf.reduce_sum(query_embs * mask, 1) / tf.where(tf.less(div, 1e-7), div+1, div)],
            1)

    elif model.hparams.user_struct == 'none':
        asin_idxs_list = model.model.history_asin_idxs
        asin_vecs, r_terms = get_asin_vec(model, asin_idxs_list, True)
        user_vec = tf.zeros_like(tf.reduce_sum(asin_vecs, 1))

    elif model.hparams.user_struct == 'user_abstract':
        user_vec = tf.nn.embedding_lookup(model.user_emb, model.user_idxs)
        combined_emb = tf.concat([user_vec, query_vec], 1)
        user_vec, r_terms = dnn_abstract(combined_emb, model.hparams.embed_size, reuse,
                                                     'user_query_abstract', tf.nn.elu)
        regularization_terms.extend(r_terms)

    elif model.hparams.user_struct == 'user_attention':
        user_vec = tf.nn.embedding_lookup(model.user_emb, model.user_idxs)
        user_vec = tf.expand_dims(user_vec, 1)
        batch_size = tf.shape(user_vec)[0]
        with variable_scope.variable_scope(scope or 'user_attention', reuse=reuse):
            user_vec = compute_attention_vec(query_vec, user_vec, tf.ones([batch_size]), True)
        regularization_terms.append(user_vec)

    else:
        sys.exit('Error! No user embedding method named %s' % model.hparams.user_struct)
    # print(user_vec.get_shape())
    #user_vec = tf.Print(user_vec, [user_vec], 'this is user_vec', summarize=50)
    return user_vec, regularization_terms

def dnn_abstract(input_vec, output_dims, reuse, scope=None, activation_func=tf.nn.tanh):
    sys.exit("Have not yet implement the function, dnn_abstract!")

def get_embedding_from_words(model, word_idxs, word_lengths, embed_func, reuse, scope=None):
    print("Have not yet implement the function")
    sys.exit("Have not yet implement the function, get_embedding_from_words")

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
        for a in xrange(num_heads):
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

def UQP_nce_loss(model, user_idxs, query_word_idxs, product_idxs, review_idxs, word_idxs, context_word_idxs = None):

    batch_size = array_ops.shape(word_idxs)[0]#get batch_size
    loss = None
    regularization_terms = []
    # user, product embedding
    if model.need_review:
        #review prediction loss
        ur_loss, ur_embs = single_nce_loss(model,user_idxs, model.user_emb, review_idxs, model.review_emb,
                        model.review_bias, model.review_size, model.review_distribute)
        pr_loss, pr_embs = single_nce_loss(model,product_idxs, model.product_emb, review_idxs, model.review_emb,
                        model.review_bias, model.review_size, model.review_distribute)
        #word prediction loss
        rw_loss, rw_embs = single_nce_loss(model,review_idxs, model.review_emb, word_idxs, model.word_emb,
                        model.word_bias, model.vocab_size, model.vocab_distribute)
        loss = ur_loss + pr_loss + rw_loss
        regularization_terms += ur_embs + pr_embs + rw_embs
    else:
        #word prediction loss
        uw_loss, uw_embs = single_nce_loss(model,user_idxs, model.user_emb, word_idxs, model.word_emb,
                        model.word_bias, model.vocab_size, model.vocab_distribute)
        pw_loss, pw_embs = single_nce_loss(model,product_idxs, model.product_emb, word_idxs, model.word_emb,
                        model.word_bias, model.vocab_size, model.vocab_distribute)
        loss = uw_loss + pw_loss
        regularization_terms += uw_embs + pw_embs

    # context prediction loss
    if model.need_context:
        for context_word_idx in context_word_idxs:
            wc_loss, wc_embs = single_nce_loss(model,word_idxs, model.word_emb, context_word_idx, model.context_emb,
                        model.context_bias, model.vocab_size, model.vocab_distribute)
            loss += wc_loss
            regularization_terms += wc_embs
    # get query vector
    query_vec, qw_embs = get_query_embedding(model, query_word_idxs, None)
    regularization_terms += qw_embs

    #product prediction loss
    uqr_loss, uqr_embs = pair_search_loss(model, query_vec, user_idxs, model.user_emb, product_idxs, model.product_emb,
                        model.product_bias, model.product_size, model.product_distribute)
    regularization_terms += uqr_embs
    loss += uqr_loss

    # L2 regularization
    if model.L2_lambda > 0:
        l2_loss = tf.nn.l2_loss(regularization_terms[0])
        for i in xrange(1,len(regularization_terms)):
            l2_loss += tf.nn.l2_loss(regularization_terms[i])
        loss += model.L2_lambda * l2_loss

    return loss / math_ops.cast(batch_size, dtypes.float32)

def pair_search_loss(model, query_vec, example_idxs, example_emb, label_idxs, label_emb,label_bias, label_size, label_distribution):

    batch_size = array_ops.shape(example_idxs)[0]#get batch_size
    # Nodes to compute the nce loss w/ candidate sampling.
    labels_matrix = tf.reshape(tf.cast(label_idxs,dtype=tf.int64),[batch_size, 1])

    # Negative sampling.
    sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=model.negative_sample,
            unique=False,
            range_max=label_size,
            distortion=0.75,
            unigrams=label_distribution))

    #get example embeddings [batch_size, embed_size]
    example_vec = tf.nn.embedding_lookup(example_emb, example_idxs) * (1-model.Wu) + query_vec * model.Wu

    #get label embeddings and bias [batch_size, embed_size], [batch_size, 1]
    true_w = tf.nn.embedding_lookup(label_emb, label_idxs)
    true_b = tf.nn.embedding_lookup(label_bias, label_idxs)

    #get sampled embeddings and bias [num_sampled, embed_size], [num_sampled, 1]
    sampled_w = tf.nn.embedding_lookup(label_emb, sampled_ids)
    sampled_b = tf.nn.embedding_lookup(label_bias, sampled_ids)

    # True logits: [batch_size, 1]
    true_logits = tf.reduce_sum(tf.multiply(example_vec, true_w), 1) + true_b

    # Sampled logits: [batch_size, num_sampled]
    # We replicate sampled noise lables for all examples in the batch
    # using the matmul.
    sampled_b_vec = tf.reshape(sampled_b, [model.negative_sample])
    sampled_logits = tf.matmul(example_vec, sampled_w, transpose_b=True) + sampled_b_vec

    return nce_loss(model, true_logits, sampled_logits), [example_vec, true_w, sampled_w]


def single_nce_loss(model, example_idxs, example_emb, label_idxs, label_emb, label_bias, label_size, label_distribution):

    batch_size = array_ops.shape(example_idxs)[0]#get batch_size
    # Nodes to compute the nce loss w/ candidate sampling.
    labels_matrix = tf.reshape(tf.cast(label_idxs,dtype=tf.int64),[batch_size, 1])

    # Negative sampling.
    sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=model.negative_sample,
            unique=False,
            range_max=label_size,
            distortion=0.75,
            unigrams=label_distribution))

    #get example embeddings [batch_size, embed_size]
    example_vec = tf.nn.embedding_lookup(example_emb, example_idxs)

    #get label embeddings and bias [batch_size, embed_size], [batch_size, 1]
    true_w = tf.nn.embedding_lookup(label_emb, label_idxs)
    true_b = tf.nn.embedding_lookup(label_bias, label_idxs)

    #get sampled embeddings and bias [num_sampled, embed_size], [num_sampled, 1]
    sampled_w = tf.nn.embedding_lookup(label_emb, sampled_ids)
    sampled_b = tf.nn.embedding_lookup(label_bias, sampled_ids)

    # True logits: [batch_size, 1]
    true_logits = tf.reduce_sum(tf.multiply(example_vec, true_w), 1) + true_b

    # Sampled logits: [batch_size, num_sampled]
    # We replicate sampled noise lables for all examples in the batch
    # using the matmul.
    sampled_b_vec = tf.reshape(sampled_b, [model.negative_sample])
    sampled_logits = tf.matmul(example_vec, sampled_w, transpose_b=True) + sampled_b_vec

    return nce_loss(model, true_logits, sampled_logits), [example_vec, true_w, sampled_w]
    #return model.nce_loss(true_logits, true_logits)

def single_nce_loss_with_vec(model, example_vec, label_idxs, label_emb, label_bias, label_size, label_distribution):
    """Combine nce loss with negative samples.
    Args:
        model: the model that contains all the functions and parameters.
        example_vec: the example vectors.
        label_idxs: the index of the label vectors.
        label_emb: the embedding matrix of the example vectors.
        label_bias: the bias matrix of the example vectors.
        label_size: the number of all possible labels.
        label_distribution: the distribution used for sampling labels.

    Returns:
        Aggregated nce loss and the parameters for L2 regularization.

    """
    batch_size = array_ops.shape(label_idxs)[0]  # get batch_size
    # Nodes to compute the nce loss w/ candidate sampling.
    labels_matrix = tf.reshape(tf.cast(label_idxs, dtype=tf.int64), [batch_size, 1])

    # Negative sampling.
    sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels_matrix,
        num_true=1,
        num_sampled=model.negative_sample,
        unique=False,
        range_max=label_size,
        distortion=0.75,
        unigrams=label_distribution))

    # get label embeddings and bias [batch_size, embed_size], [batch_size, 1]
    true_w = tf.nn.embedding_lookup(label_emb, label_idxs)
    true_b = tf.nn.embedding_lookup(label_bias, label_idxs)

    # get sampled embeddings and bias [num_sampled, embed_size], [num_sampled, 1]
    sampled_w = tf.nn.embedding_lookup(label_emb, sampled_ids)
    sampled_b = tf.nn.embedding_lookup(label_bias, sampled_ids)

    # True logits: [batch_size, 1]
    true_logits = tf.reduce_sum(tf.multiply(example_vec, true_w), 1) + true_b

    # Sampled logits: [batch_size, num_sampled]
    # We replicate sampled noise lables for all examples in the batch
    # using the matmul.
    sampled_b_vec = tf.reshape(sampled_b, [model.negative_sample])
    sampled_logits = tf.matmul(example_vec, sampled_w, transpose_b=True) + sampled_b_vec

    return nce_loss(model, true_logits, sampled_logits), [example_vec, true_w, sampled_w]

def nce_loss(model, true_logits, sampled_logits):
    "Build the graph for the NCE loss."

    # cross-entropy(logits, labels)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=true_logits, labels=tf.ones_like(true_logits))
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=sampled_logits, labels=tf.zeros_like(sampled_logits))

    # NCE-loss is the sum of the true and noise (sampled words)
    # contributions, averaged over the batch.
    nce_loss_tensor = (tf.reduce_sum(true_xent) + tf.reduce_sum(sampled_xent))
    return nce_loss_tensor
