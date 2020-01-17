from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import range# pylint: disable=redefined-builtin
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
import logging

import numpy as np
from six.moves import range# pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf

from esrt.query_embedding import get_query_embedding
from esrt.losses import pair_search_loss, relation_nce_loss
from esrt.engine.base_model import BaseModel


class DREM(BaseModel):
    def __init__(self, dataset, params, forward_only=False):
        """Create the self.

        Args:
            vocab_size: the number of words in the corpus.
            dm_feature_len: the length of document model features (query based).
            review_size: the number of reviews in the corpus.
            user_size: the number of users in the corpus.
            product_size: the number of products in the corpus.
            embed_size: the size of each embedding
            window_size: the size of half context window
            vocab_distribute: the distribution for words, used for negative sampling
            review_distribute: the distribution for reviews, used for negative sampling
            product_distribute: the distribution for products, used for negative sampling
            max_gradient_norm: gradients will be clipped to maximally this norm.
            batch_size: the size of the batches used during training;
            the model construction is not independent of batch_size, so it cannot be
            changed after initialization.
            learning_rate: learning rate to start with.
            learning_rate_decay_factor: decay learning rate by this much when needed.
            forward_only: if set, we do not construct the backward pass in the self.
            negative_sample: the number of negative_samples for training
        """
        self._dataset = dataset

        self._params = params
        self.negative_sample = self._params['negative_sample']
        self.embed_size = self._params['embed_size']
        self.window_size = self._params['window_size']
        self.max_gradient_norm = self._params['max_gradient_norm']
        self.init_learning_rate = self._params['init_learning_rate']
        self.L2_lambda = self._params['L2_lambda']
        self.net_struct = self._params['net_struct']
        self.similarity_func = self._params['similarity_func']
        self.dynamic_weight = self._params['dynamic_weight']
        self.query_weight= self._params['query_weight']
        self.global_step = tf.Variable(0, trainable=False)
        if self.query_weight >= 0:
            self.Wu = tf.Variable(self.query_weight, name="user_weight", dtype=tf.float32, trainable=False)
        else:
            self.Wu = tf.sigmoid(tf.Variable(0, name="user_weight", dtype=tf.float32))
        self.query_max_length = dataset.query_max_length

        self.forward_only = forward_only

        #self.print_ops = []

        print('L2 lambda ' + str(self.L2_lambda))

    def entity(self, name, vocab):
        """
        Create a 'struct' for entity node.

        Args:
            name: str
            vocab: List with shape of [len(vocab)] with type of str, is a map from idx -> id.
        """
        init_width = 0.5 / self.embed_size
        print('%s size %s' % (name,str(len(vocab))))
        return {
            'name' : name,
            'vocab' : vocab,
            'size' : len(vocab),
            'embedding' :tf.Variable( tf.random_uniform(
                        [len(vocab) + 1, self.embed_size], -init_width, init_width),
                        name="%s_emb"%name)
        }

    def relation(self, name, distribute, tail_entity):
        """
        Create a 'struct' for relation edge.

        Args:
            name: str.
            distribute: List with shape of [len(vocab(tail_entity))] with type of int.
            tail_entity: str, represent 'entity struct'.
        """
        print('%s size %s' % (name, str(len(distribute))))
        init_width = 0.5 / self.embed_size
        return {
            'name' : name,
            'tail_entity' : tail_entity,
            'distribute' : distribute,
            'idxs' : tf.placeholder(tf.int64, shape=[None], name="%s_idxs"%name),
            'weight' :     tf.placeholder(tf.float32, shape=[None], name="%s_weight"%name),
            'embedding' : tf.Variable( tf.random_uniform(
                            [self.embed_size], -init_width, init_width),
                            name="%s_emb"%name),
            'bias' : tf.Variable(tf.zeros([len(distribute)+1]), name="%s_b"%name)
        }

    def build(self):
        self._build_placeholder()
        self.loss = self._build_embedding_graph_and_loss()
        if not self.forward_only:
            self.updates = self._build_optimizer()
        else:
            self.product_scores, uq_vec = self.get_product_scores(self.user_idxs, self.query_word_idxs)

        self.saver = tf.train.Saver(tf.global_variables())

    def _build_placeholder(self):
        self.query_word_idxs = tf.placeholder(tf.int64, shape=[None, self.query_max_length], name="query_word_idxs")
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        self.user_idxs = tf.placeholder(tf.int64, shape=[None], name="user_idxs")
        self.product_idxs = tf.placeholder(tf.int64, shape=[None], name="product_idxs")

        self.entity_dict = {
            'user' : self.entity('user', self._dataset.user_ids),
            'product' : self.entity('product', self._dataset.product_ids),
            'word' : self.entity('word', self._dataset.words),
            'related_product' : self.entity('related_product',self._dataset.related_product_ids),
            'brand' : self.entity('brand', self._dataset.brand_ids),
            'categories' : self.entity('categories', self._dataset.category_ids),
        }

        self.relation_dict = {
            'word' : self.relation('write', self._dataset.vocab_distribute, 'word'),
            'also_bought' : self.relation('also_bought', self._dataset.knowledge['also_bought']['distribute'], 'related_product'),
            'also_viewed' : self.relation('also_viewed', self._dataset.knowledge['also_viewed']['distribute'], 'related_product'),
            'bought_together' : self.relation('bought_together', self._dataset.knowledge['bought_together']['distribute'], 'related_product'),
            'brand' : self.relation('is_brand', self._dataset.knowledge['brand']['distribute'], 'brand'),
            'categories' : self.relation('is_category', self._dataset.knowledge['categories']['distribute'], 'categories')
        }


    def _build_embedding_graph_and_loss(self, scope=None):
        # decide which relation we want to use
        self.use_relation_dict = {
            'also_bought' : False,
            'also_viewed' : False,
            'bought_together' : False,
            'brand' : False,
            'categories' : False,
        }
        if 'none' in self.net_struct:
            print('Use no relation')
        else:
            need_relation_list = []
            for key in self.use_relation_dict:
                if key in self.net_struct:
                    self.use_relation_dict[key] = True
                    need_relation_list.append(key)
            if len(need_relation_list) > 0:
                print('Use relation ' + ' '.join(need_relation_list))
            else:
                print('Use all relation')
                for key in self.use_relation_dict:
                    self.use_relation_dict[key] = True

        # build graph
        with variable_scope.variable_scope(scope or "embedding_graph"):
            loss = None
            regularization_terms = []
            batch_size = array_ops.shape(self.user_idxs)[0]#get batch_size
            # user + query -> product
            query_vec = None
            if self.dynamic_weight >= 0.0:
                print('Treat query as a dynamic relationship')
                query_vec, qw_embs = get_query_embedding(self, self.query_word_idxs, self.entity_dict['word']['embedding'], None) # get query vector
                regularization_terms.extend(qw_embs)
            else:
                print('Treat query as a static relationship')
                init_width = 0.5 / self.embed_size
                self.query_static_vec = tf.Variable(tf.random_uniform([self.embed_size], -init_width, init_width),
                                    name="query_emb")
                query_vec = self.query_static_vec
                regularization_terms.extend([query_vec])
            self.product_bias = tf.Variable(tf.zeros([self.entity_dict['product']['size'] + 1]), name="product_b")
            uqr_loss_tensor, uqr_embs = pair_search_loss(self, self.Wu, query_vec, self.user_idxs, # product prediction loss
                                self.entity_dict['user']['embedding'], self.product_idxs,
                                self.entity_dict['product']['embedding'], self.product_bias,
                                len(self.entity_dict['product']['vocab']), self._dataset.product_distribute)
            regularization_terms.extend(uqr_embs)

            dynamic_loss = tf.reduce_sum(uqr_loss_tensor)
            #self.print_ops.append(tf.print('dynamic_loss: ', dynamic_loss, '\n'))

            # user + write -> word
            uw_loss_tensor, uw_embs = relation_nce_loss(self, 0.5, self.user_idxs, 'user', 'word', 'word')
            regularization_terms.extend(uw_embs)
            #self.print_ops.append(tf.print('uw_loss: ', tf.reduce_sum(uw_loss_tensor), '\n'))

            static_loss = tf.reduce_sum(uw_loss_tensor)

            # product + write -> word
            pw_loss_tensor, pw_embs = relation_nce_loss(self, 0.5, self.product_idxs, 'product', 'word', 'word')
            regularization_terms.extend(pw_embs)
            #self.print_ops.append(tf.print('pw_loss: ', tf.reduce_sum(pw_loss_tensor), '\n'))
            static_loss += tf.reduce_sum(pw_loss_tensor)

            # product + also_bought -> product
            if self.use_relation_dict['also_bought']:
                pab_loss_tensor, pab_embs = relation_nce_loss(self, 0.5, self.product_idxs, 'product', 'also_bought', 'related_product')
                regularization_terms.extend(pab_embs)
                #self.print_ops.append(tf.print('pab_loss: ', tf.reduce_sum(pab_loss_tensor), '\n'))
                static_loss += tf.reduce_sum(pab_loss_tensor)

            # product + also_viewed -> product
            if self.use_relation_dict['also_viewed']:
                pav_loss_tensor, pav_embs = relation_nce_loss(self, 0.5, self.product_idxs, 'product', 'also_viewed', 'related_product')
                regularization_terms.extend(pav_embs)
                #self.print_ops.append(tf.print('pav_loss: ', tf.reduce_sum(pav_loss_tensor), '\n'))
                static_loss += tf.reduce_sum(pav_loss_tensor)

            # product + bought_together -> product
            if self.use_relation_dict['bought_together']:
                pbt_loss_tensor, pbt_embs = relation_nce_loss(self, 0.5, self.product_idxs, 'product', 'bought_together', 'related_product')
                regularization_terms.extend(pbt_embs)
                #self.print_ops.append(tf.print('pbt_loss: ', tf.reduce_sum(pbt_loss_tensor), '\n'))
                static_loss += tf.reduce_sum(pbt_loss_tensor)

            # product + is_brand -> brand
            if self.use_relation_dict['brand']:
                pib_loss_tensor, pib_embs = relation_nce_loss(self, 0.5, self.product_idxs, 'product', 'brand', 'brand')
                regularization_terms.extend(pib_embs)
                #self.print_ops.append(tf.print('pib_loss: ', tf.reduce_sum(pib_loss_tensor), '\n'))
                static_loss += tf.reduce_sum(pib_loss_tensor)

            # product + is_category -> categories
            if self.use_relation_dict['categories']:
                pic_loss_tensor, pic_embs = relation_nce_loss(self, 0.5, self.product_idxs, 'product', 'categories', 'categories')
                regularization_terms.extend(pic_embs)
                #self.print_ops.append(tf.print('pic_loss: ', tf.reduce_sum(pic_loss_tensor), '\n'))
                static_loss += tf.reduce_sum(pic_loss_tensor)


            #self.print_ops.append(tf.print('satic_loss: ', static_loss, '\n======\n'))
            # merge dynamic loss and static loss
            loss = None
            if self.dynamic_weight >= 0.0:
                print('Dynamic relation weight %.2f' % self.dynamic_weight)
                loss = 2 * (self.dynamic_weight * dynamic_loss + (1-self.dynamic_weight) * static_loss)
            else:
                # consider query as a static relation
                loss = dynamic_loss + static_loss

            # L2 regularization
            if self.L2_lambda > 0:
                l2_loss = tf.nn.l2_loss(regularization_terms[0])
                for i in range(1,len(regularization_terms)):
                    l2_loss += tf.nn.l2_loss(regularization_terms[i])
                loss += self.L2_lambda * l2_loss

            return loss / math_ops.cast(batch_size, dtypes.float32)

    def _build_optimizer(self):
        params = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.gradients = tf.gradients(self.loss, params)

        self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
                                                                 self.max_gradient_norm)
        return opt.apply_gradients(zip(self.clipped_gradients, params),
                                         global_step=self.global_step)

    def step(self, session, input_feed, forward_only, file_writer=None, test_mode = 'product_scores'):
        """Run a step of the model feeding the given inputs.

        Args:
            session: tensorflow session to use.
            learning_rate: the learning rate of current step
            user_idxs: A numpy [1] float vector.
            product_idxs: A numpy [1] float vector.
            review_idxs: A numpy [1] float vector.
            word_idxs: A numpy [None] float vector.
            context_idxs: list of numpy [None] float vectors.
            forward_only: whether to do the update step or only forward.

        Returns:
            A triple consisting of gradient norm (or None if we did not do backward),
            average perplexity, and the outputs.

        Raises:
            ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """

        # Output feed: depends on whether we do a backward step or not.
        entity_list = None
        if not forward_only:
            output_feed = [self.updates,    # Update Op that does SGD.
                            self.loss]     # Loss for this batch.
                            #self.print_ops]
        else:
            if test_mode == 'output_embedding':
                self.embed_output_keys = []
                output_feed = []
                for key in self.entity_dict:
                    self.embed_output_keys.append(key)
                    output_feed.append(self.entity_dict[key]['embedding'])
                for key in self.relation_dict:
                    self.embed_output_keys.append(key + '_embed')
                    output_feed.append(self.relation_dict[key]['embedding'])
                for key in self.relation_dict:
                    self.embed_output_keys.append(key + '_bias')
                    output_feed.append(self.relation_dict[key]['bias'])
                self.embed_output_keys.append('Wu')
                output_feed.append(self.Wu)
            elif 'explain' in test_mode:
                if test_mode == 'explain_user_query':
                    entity_list = self.uq_entity_list
                elif test_mode == 'explain_product':
                    entity_list = self.p_entity_list
                output_feed = [scores for _, _, scores in entity_list]
            else:
                output_feed = [self.product_scores] #negative instance output

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1]
        else:
            if test_mode == 'output_embedding':
                return outputs, self.embed_output_keys
            elif 'explain' in test_mode:
                return [(entity_list[i][0], entity_list[i][1], outputs[i]) for i in range(len(entity_list))], None
            else:
                return outputs[0], None    # product scores to input user

    def get_product_scores(self, user_idxs, query_word_idx, product_idxs = None, scope = None):
    	with variable_scope.variable_scope(scope or "embedding_graph"):
    		# get user embedding [None, embed_size]
    		user_vec = tf.nn.embedding_lookup(self.entity_dict['user']['embedding'], user_idxs)
    		# get query embedding [None, embed_size]
    		if self.dynamic_weight >= 0.0:
    			print('Query as a dynamic relationship')
    			query_vec, query_embs = get_query_embedding(self, query_word_idx, self.entity_dict['word']['embedding'], True)
    		else:
    			print('Query as a static relationship')
    			query_vec = self.query_static_vec

    		# get candidate product embedding [None, embed_size]
    		product_vec = None
    		product_bias = None
    		if product_idxs != None:
    			product_vec = tf.nn.embedding_lookup(self.entity_dict['product']['embedding'], product_idxs)
    			product_bias = tf.nn.embedding_lookup(self.product_bias, product_idxs)
    		else:
    			product_vec = self.entity_dict['product']['embedding']
    			product_bias = self.product_bias

    		print('Similarity Function : ' + self.similarity_func)
    		example_vec = (1.0 - self.Wu) * user_vec + self.Wu * query_vec
    		#example_vec = user_vec + query_vec

    		if self.similarity_func == 'product':
    			return tf.matmul(example_vec, product_vec, transpose_b=True), example_vec
    		elif self.similarity_func == 'bias_product':
    			return tf.matmul(example_vec, product_vec, transpose_b=True) + product_bias, example_vec
    		else:
    			norm_vec = example_vec / tf.sqrt(tf.reduce_sum(tf.square(example_vec), 1, keep_dims=True))
    			product_vec = product_vec / tf.sqrt(tf.reduce_sum(tf.square(product_vec), 1, keep_dims=True))
    			return tf.matmul(norm_vec, product_vec, transpose_b=True), example_vec
