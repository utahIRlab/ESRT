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
import tensorflow as tf




class QueryEmbedding:
    @staticmethod
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

    @staticmethod
    def get_addition_from_words(model, word_idxs, reuse, scope=None):
        with variable_scope.variable_scope(scope or 'addition_abstraction',
                                             reuse=reuse):
            # get mean word vectors
            word_vecs = tf.nn.embedding_lookup(model.word_emb, word_idxs)
            mean_word_vec = tf.reduce_mean(word_vecs, 1)
            return mean_word_vec, [word_vecs]

    @staticmethod
    def get_RNN_from_words(model, word_idxs, reuse, scope=None):
        with variable_scope.variable_scope(scope or 'RNN_abstraction',
                                             reuse=reuse):
            # get mean word vectors
            word_vecs = tf.nn.embedding_lookup(model.word_emb, word_idxs)
            cell = tf.contrib.rnn.GRUCell(model.embed_size)
            encoder_outputs, encoder_state = tf.nn.static_rnn(cell, tf.unstack(word_vecs, axis=1), dtype=dtypes.float32)
            return encoder_state, [word_vecs]

    @staticmethod
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
    get_attention_from_words = QueryEmbedding.get_attention_from_words

    print('Query model: Attention')
    return get_attention_from_words(model, word_idxs, reuse, scope)

def UQP_nce_loss(model, user_idxs, query_word_idxs, product_idxs, review_idxs,
                            word_idxs, context_word_idxs = None):
    batch_size = array_ops.shape(word_idxs)[0]#get batch_size
    loss = None
    regularization_terms = []

    #word prediction loss
    uw_loss, uw_embs = single_nce_loss(model,user_idxs, model.user_emb, word_idxs, model.word_emb,
                    model.word_bias, model.vocab_size, model.vocab_distribute)
    pw_loss, pw_embs = single_nce_loss(model,product_idxs, model.product_emb, word_idxs, model.word_emb,
                    model.word_bias, model.vocab_size, model.vocab_distribute)
    loss = uw_loss + pw_loss
    regularization_terms += uw_embs + pw_embs

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



def pair_search_loss(model, query_vec, example_idxs, example_emb, label_idxs, label_emb,
                    label_bias, label_size, label_distribution):
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

def single_nce_loss(model, example_idxs, example_emb, label_idxs, label_emb,
                    label_bias, label_size, label_distribution):
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
