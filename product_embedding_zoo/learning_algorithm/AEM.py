# 3rd party import
import tensorflow.compat.v1 as tf
from six.moves import xrange

# strandard library
import logging

# package import
from . import AEM_builder
from . import LSE
import product_embedding_zoo as pez

class AEM():
    def __init__(self, dataset, hparams, forward_only=False):
        # dataset paramters
        self.dataset = dataset
        self.vocab_size = self.dataset.vocab_size
        self.review_size = self.dataset.review_size
        self.user_size = self.dataset.user_size
        self.product_size = self.dataset.product_size
        self.query_max_length = self.dataset.query_max_length
        self.vocab_distribute = self.dataset.vocab_distribute
        self.review_distribute = self.dataset.review_distribute
        self.product_distribute = self.dataset.product_distribute

        # hparams setting
        self.hparams = hparams
        self.negative_sample = self.hparams.negative_sample
        self.embed_size = self.hparams.embed_size
        self.window_size = self.hparams.window_size
        self.max_gradient_norm = self.hparams.max_gradient_norm
        #self.batch_size = batch_size * (self.negative_sample + 1)
        self.init_learning_rate = self.hparams.init_learning_rate
        self.L2_lambda = self.hparams.L2_lambda
        self.net_struct = self.hparams.net_struct
        self.similarity_func = self.hparams.similarity_func
        self.query_weight=self.hparams.query_weight
        self.max_history_length = min(self.hparams.max_history_length, self.dataset.max_history_length)
        self.global_step = tf.Variable(0, trainable=False)
        if self.query_weight >= 0:
            self.Wu = tf.Variable(self.query_weight, name="user_weight", dtype=tf.float32, trainable=False)
        else:
            self.Wu = tf.sigmoid(tf.Variable(0, name="user_weight", dtype=tf.float32))

        self.print_ops = [] # for debug
        # hparams debugger
        logging.info(
            """
                negative_sample: %d
                embed_size: %d
                window_size: %d
                max_gradient_norm: %f
                init_learning_rate: %f
                L2_lambda: %f
                net_struct: %s
                similarity_func: %s
                query_weight: %s
            """%(self.negative_sample,
                 self.embed_size,
                 self.window_size,
                 self.max_gradient_norm,
                 self.init_learning_rate,
                 self.L2_lambda,
                 self.net_struct,
                 self.similarity_func,
                 self.query_weight)
        )

        # create placeholders
        self._create_placeholder()

        # specify model structure
        logging.info("Model Name " + self.net_struct)
        self.need_review = True
        if 'simplified' in self.net_struct:
            print('Simplified model')
            self.need_review = False

        self.need_context = False
        if 'hdc' in self.net_struct:
            print('Use context words')
            self.need_context = True

        if 'LSE' == self.net_struct:
            self.need_review = False
            self.need_context = True

        if self.need_context:
            self.context_word_idxs = []
            for i in xrange(2 * self.window_size):
                self.context_word_idxs.append(tf.placeholder(tf.int64, shape=[None], name="context_idx{0}".format(i)))

        # Training losses.
        self.loss = None
        if 'LSE' == self.net_struct:
            self.loss = LSE.build_embedding_graph_and_loss(self)
        else:
            self.loss = AEM_builder.build_embedding_graph_and_loss(self)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.gradients = tf.gradients(self.loss, params)

            self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
                                                                     self.max_gradient_norm)
            self.updates = opt.apply_gradients(zip(self.clipped_gradients, params),
                                             global_step=self.global_step)

            #self.updates = opt.apply_gradients(zip(self.gradients, params),
            #                                 global_step=self.global_step)
        else:
            if 'LSE' == self.net_struct:
                self.product_scores = LSE.get_product_scores(self, self.query_word_idxs)
            else:
                self.product_scores = AEM_builder.get_product_scores(self, self.user_idxs, self.query_word_idxs)

        # Add tf.summary scalar
        tf.summary.scalar('Learning_rate', self.learning_rate, collections=['train'])
        tf.summary.scalar('Loss', self.loss, collections=['train'])
        self.train_summary = tf.summary.merge_all(key='train')

        self.saver = tf.train.Saver(tf.global_variables())

    def _create_placeholder(self):
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

    def step(self, session, input_feed, forward_only, file_writer=None, test_mode='product_scores'):
        if not forward_only:
            output_feed = [self.updates,    # Update Op that does SGD.
                         self.loss,
                         self.train_summary]
                         #self.print_ops]    # Loss for this batch.
        else:
            if test_mode == 'output_embedding':
                output_feed = [self.user_emb, self.product_emb, self.Wu, self.word_emb, self.word_bias]
                if self.need_review:
                    output_feed += [self.review_emb, self.review_bias]

                if self.need_context and 'LSE' != self.net_struct:
                    output_feed += [self.context_emb, self.context_bias]

            else:
                output_feed = [self.product_scores] #negative instance output

        # debug
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
        #print("This is the summary", self.train_summary)
        outputs = session.run(output_feed, input_feed)  #options=run_options, run_metadata=run_metadata)
        #file_writer.add_run_metadata(run_metadata,"step%d"%session.run(self.global_step))
        #loss, train_summary = outputs[1], outputs[-1]
        #print("loss is ", loss, "summary is: ", train_summary, outputs)
        if not forward_only:
            return outputs[1], outputs[2]   # loss, no outputs, Gradient norm.
        else:
            if test_mode == 'output_embedding':
                return outputs[:4], outputs[4:]
            else:
                return outputs[0], None    # product scores to input user
