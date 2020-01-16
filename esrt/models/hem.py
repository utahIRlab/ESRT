

import tensorflow.compat.v1 as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


from esrt.engine.base_model import BaseModel
from esrt.losses import single_nce_loss, pair_search_loss
from esrt.query_embedding import get_query_embedding


class HEM(BaseModel):
    def __init__(self, dataset, params, forward_only=False):
        self._dataset = dataset
        self.vocab_size = self._dataset.vocab_size
        self.review_size = self._dataset.review_size
        self.user_size = self._dataset.user_size
        self.product_size = self._dataset.product_size
        self.query_max_length = self._dataset.query_max_length
        self.vocab_distribute = self._dataset.vocab_distribute
        self.review_distribute = self._dataset.review_distribute
        self.product_distribute = self._dataset.product_distribute

        self._params = params
        self.negative_sample = self._params['negative_sample']
        self.embed_size = self._params['embed_size']
        self.window_size = self._params['window_size']
        self.max_gradient_norm = self._params['max_gradient_norm']
        self.init_learning_rate = self._params['init_learning_rate']
        self.L2_lambda = self._params['L2_lambda']
        self.net_struct = self._params['net_struct']
        self.similarity_func = self._params['similarity_func']
        self.query_weight=self._params['query_weight']
        self.global_step = tf.Variable(0, trainable=False)

        self.forward_only = forward_only

        self.print_ops = []
        if self.query_weight >= 0:
            self.Wu = tf.Variable(self.query_weight, name="user_weight", dtype=tf.float32, trainable=False)
        else:
            self.Wu = tf.sigmoid(tf.Variable(0, name="user_weight", dtype=tf.float32))


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
        self.word_idxs = tf.placeholder(tf.int64, shape=[None], name="word_idxs")
        self.query_word_idxs = tf.placeholder(tf.int64, shape=[None, self.query_max_length], name="query_word_idxs")
        self.PAD_embed = tf.get_variable("PAD_embed", [1,self.embed_size],dtype=tf.float32)

    def _build_embedding_graph_and_loss(self, scope=None):
        with variable_scope.variable_scope(scope or "embedding_graph"):
            # define all variables
            init_width = 0.5 / self.embed_size
            self.word_emb = tf.Variable( tf.random_uniform(
                                [self.vocab_size+1, self.embed_size], -init_width, init_width),
                                name="word_emb")
            self.word_bias = tf.Variable(tf.zeros([self.vocab_size+1]), name="word_b")
            # user/product embeddings.
            self.user_emb =    tf.Variable( tf.zeros([self.user_size, self.embed_size]),
                                name="user_emb")
            self.user_bias =    tf.Variable( tf.zeros([self.user_size]), name="user_b")
            self.product_emb =    tf.Variable( tf.zeros([self.product_size, self.embed_size]),
                                name="product_emb")
            self.product_bias =    tf.Variable( tf.zeros([self.product_size]), name="product_b")

            # define computation graph
            batch_size = array_ops.shape(self.word_idxs)[0]
            loss = None
            regularization_terms = []

            # predict words loss
            uw_loss, uw_embs = single_nce_loss(self, self.user_idxs, self.user_emb, self.word_idxs, self.word_emb,
                            self.word_bias, self.vocab_size, self.vocab_distribute)
            pw_loss, pw_embs = single_nce_loss(self, self.product_idxs, self.product_emb, self.word_idxs, self.word_emb,
                            self.word_bias, self.vocab_size, self.vocab_distribute)
            loss = uw_loss + pw_loss
            regularization_terms += uw_embs + pw_embs

            # pair search loss
            query_vec, qw_embs = get_query_embedding(self, self.query_word_idxs, None)
            regularization_terms += qw_embs

            uqr_loss, uqr_embs = pair_search_loss(self, query_vec, self.user_idxs, self.user_emb, self.product_idxs, self.product_emb,
                                self.product_bias, self.product_size, self.product_distribute)
            regularization_terms += uqr_embs
            loss += uqr_loss

            # regulaizer loss
            if self.L2_lambda > 0:
                l2_loss = tf.nn.l2_loss(regularization_terms[0])
                for i in range(1,len(regularization_terms)):
                    l2_loss += tf.nn.l2_loss(regularization_terms[i])

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
            else:
                output_feed = [self.product_scores, self.print_ops]

        outputs = session.run(output_feed, input_feed)  #options=run_options, run_metadata=run_metadata)

        if not forward_only:
            return outputs[1]   # loss, no outputs, Gradient norm.
        else:
            if test_mode == 'output_embedding':
                return outputs[:4], outputs[4:]
            else:
                return outputs[0], None    # product scores to input user

    def get_product_scores(self, user_idxs, query_word_idx, product_idxs = None, scope = None):
        """
        Args:
            user_idxs: Tensor with shape of [batch_size] with type of int32.
            query_word_idx: Tensor with shape for [batch_size, query_max_length] with type of int32.
            product_idxs: Tensor with shape of [batch_size] with type of int32 or None.
            scope:

        Return:
            product_scores: Tensor with shape of [batch_size, batch_size] or [batch_size, len(product_vocab)]
                            with type of float32. its (i, j) entry is the score of j product retrieval by i
                            example(which is a linear combination of user and query).


        """

        with variable_scope.variable_scope(scope or "embedding_graph"):
            # get user embedding [None, embed_size]
            user_vec = tf.nn.embedding_lookup(self.user_emb, user_idxs)
            # get query embedding [None, embed_size]
            query_vec, query_embs = get_query_embedding(self, query_word_idx, True)

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
