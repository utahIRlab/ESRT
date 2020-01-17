# 3rd party import
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

# strandard library
import logging

# package import
from esrt.engine.base_model import BaseModel
from esrt.user_embedding import get_user_vec
from esrt.query_embedding import get_query_embedding
from esrt.losses import single_nce_loss_with_vec, single_nce_loss

class ZAM(BaseModel):
    def __init__(self, dataset, params, forward_only=False):
        # dataset paramters
        self._dataset = dataset
        self.vocab_size = self._dataset.vocab_size
        self.review_size = self._dataset.review_size
        self.user_size = self._dataset.user_size
        self.product_size = self._dataset.product_size
        self.query_max_length = self._dataset.query_max_length
        self.vocab_distribute = self._dataset.vocab_distribute
        self.review_distribute = self._dataset.review_distribute
        self.product_distribute = self._dataset.product_distribute

        # hparams setting
        self._params = params
        self.negative_sample = self._params['negative_sample']
        self.embed_size = self._params['embed_size']
        self.window_size = self._params['window_size']
        self.max_gradient_norm = self._params['max_gradient_norm']
        #self.batch_size = batch_size * (self.negative_sample + 1)
        self.init_learning_rate = self._params['init_learning_rate']
        self.L2_lambda = self._params['L2_lambda']
        self.net_struct = self._params['net_struct']
        self.similarity_func = self._params['similarity_func']
        self.query_weight=self._params['query_weight']
        self.max_history_length = min(self._params['max_history_length'], self._dataset.max_history_length)
        self.global_step = tf.Variable(0, trainable=False)
        if self.query_weight >= 0:
            self.Wu = tf.Variable(self.query_weight, name="user_weight", dtype=tf.float32, trainable=False)
        else:
            self.Wu = tf.sigmoid(tf.Variable(0, name="user_weight", dtype=tf.float32))

        self.forward_only = forward_only

        self.print_ops = [] # for debug

        self.saver = tf.train.Saver(tf.global_variables())

    def build(self):
        self._build_placeholder()
        self.loss = self._build_embedding_graph_and_loss()

        if not self.forward_only:
            self.updates = self._build_optimizer()
        else:
            self.product_scores = self.get_product_scores(self.user_idxs, self.query_word_idxs)

        self.saver = tf.train.Saver(tf.global_variables())

    def _build_placeholder(self):
        # Feeds for inputs.
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        self.review_idxs = tf.placeholder(tf.int64, shape=[None], name="review_idxs")
        self.user_idxs = tf.placeholder(tf.int64, shape=[None], name="user_idxs")
        self.product_idxs = tf.placeholder(tf.int64, shape=[None], name="product_idxs")
        self.history_product_idxs = tf.placeholder(tf.int64, shape=[None, self.max_history_length],
                                                 name="history_product_idxs")
        self.input_history_length = tf.placeholder(tf.int64, shape=[None], name="history_length")
        self.word_idxs = tf.placeholder(tf.int64, shape=[None], name="word_idxs")
        self.query_word_idxs = tf.placeholder(tf.int64, shape=[None, self.query_max_length], name="query_word_idxs")
        self.PAD_embed = tf.get_variable("PAD_embed", [1,self.embed_size],dtype=tf.float32)

    def _build_embedding_graph_and_loss(self, scope = None):
        with variable_scope.variable_scope(scope or "embedding_graph"):
            # Word embeddings.
            init_width = 0.5 / self.embed_size
            self.word_emb = tf.Variable( tf.random_uniform(
                                [self.vocab_size+1, self.embed_size], -init_width, init_width),
                                name="word_emb")
            self.word_bias = tf.Variable(tf.zeros([self.vocab_size+1]), name="word_b")

            # user/product embeddings.
            self.user_emb =    tf.Variable( tf.zeros([self.user_size, self.embed_size]),
                                name="user_emb")
            self.user_bias =    tf.Variable( tf.zeros([self.user_size]), name="user_b")
            self.product_emb =    tf.Variable( tf.zeros([self.product_size+1, self.embed_size]),
                                name="product_emb")
            self.product_bias =    tf.Variable( tf.zeros([self.product_size+1]), name="product_b")


            loss = None
            regularization_terms = []
            pr_loss_tensor, pr_embs = single_nce_loss(self, self.product_idxs, self.product_emb, self.word_idxs, self.word_emb,
                                                self.word_bias, self.vocab_size, self.vocab_distribute)
            loss = tf.reduce_sum(pr_loss_tensor)
            regularization_terms += pr_embs
            #self.print_ops.append(tf.print("product_loss: ", pr_loss, output_stream=sys.stdout))

            # get query_vec
            query_vec, qw_embs = get_query_embedding(self, self.query_word_idxs, self.word_emb, None)
            regularization_terms += qw_embs

            # get user_vec by looking history product list
            user_vec, r_terms = get_user_vec(self, query_vec)
            regularization_terms += r_terms
            regularization_terms.append(user_vec)

            # compute the pair search loss
            combined_vec = user_vec * (1-self.Wu) + query_vec * self.Wu
            uqr_loss_tensor, uqr_embs = single_nce_loss_with_vec(self, combined_vec, self.product_idxs, self.product_emb,
                                self.product_bias, self.product_size, self.product_distribute)
            loss += tf.reduce_sum(uqr_loss_tensor)
            regularization_terms += uqr_embs

            # L2 regularization
            if self.L2_lambda > 0:
                l2_loss = tf.nn.l2_loss(regularization_terms[0])
                for i in range(1,len(regularization_terms)):
                    l2_loss += tf.nn.l2_loss(regularization_terms[i])
                loss += self.L2_lambda * l2_loss

            batch_size = array_ops.shape(self.word_idxs)[0]  #get batch_size

            return loss / math_ops.cast(batch_size, tf.float32)

    def _build_optimizer(self):
        params = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.gradients = tf.gradients(self.loss, params)

        self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
                                                                 self.max_gradient_norm)
        return opt.apply_gradients(zip(self.clipped_gradients, params),
                                         global_step=self.global_step)

    def step(self, session, input_feed, forward_only, file_writer=None, test_mode='product_scores'):
        if not forward_only:
            output_feed = [self.updates,    # Update Op that does SGD.
                         self.loss]    # Loss for this batch.
        else:
            if test_mode == 'output_embedding':
                output_feed = [self.user_emb, self.product_emb, self.Wu, self.word_emb, self.word_bias]
                if self.need_review:
                    output_feed += [self.review_emb, self.review_bias]

                if self.need_context and 'LSE' != self.net_struct:
                    output_feed += [self.context_emb, self.context_bias]

            else:
                output_feed = [self.product_scores] #negative instance output

        outputs = session.run(output_feed, input_feed)  #options=run_options, run_metadata=run_metadata)

        if not forward_only:
            return outputs[1] # loss, no outputs, Gradient norm.
        else:
            if test_mode == 'output_embedding':
                return outputs[:4], outputs[4:]
            else:
                return outputs[0], None    # product scores to input user

    def get_product_scores(self, user_idxs, query_word_idx, product_idxs = None, scope = None):
        with variable_scope.variable_scope(scope or "embedding_graph"):
            query_vec, query_embs = get_query_embedding(self, query_word_idx, self.word_emb,  True)
            user_vec, _ = get_user_vec(self, query_vec, True)

            # get candidate product embedding [None, embed_size]
            product_vec = None
            product_bias = None
            if product_idxs != None:
                product_vec = tf.nn.embedding_lookup(self.product_emb, product_idxs)
                product_bias = tf.nn.embedding_lookup(self.product_bias, product_idxs)
            else:
                product_vec = self.product_emb
                product_bias = self.product_bias

            print('Similarity Function : ' + self.similarity_func)


            if self.similarity_func == 'product':
                return tf.matmul((1.0 - self.Wu) * user_vec + self.Wu * query_vec, product_vec, transpose_b=True)
            elif self.similarity_func == 'bias_product':
                return tf.matmul((1.0 - self.Wu) * user_vec + self.Wu * query_vec, product_vec, transpose_b=True) + product_bias
            else:
                user_vec = user_vec / tf.sqrt(tf.reduce_sum(tf.square(user_vec), 1, keep_dims=True))
                query_vec = query_vec / tf.sqrt(tf.reduce_sum(tf.square(query_vec), 1, keep_dims=True))
                product_vec = product_vec / tf.sqrt(tf.reduce_sum(tf.square(product_vec), 1, keep_dims=True))
                return tf.matmul((1.0 - self.Wu) * user_vec + self.Wu * query_vec, product_vec, transpose_b=True)
