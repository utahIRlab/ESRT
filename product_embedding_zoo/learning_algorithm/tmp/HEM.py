# 3rd party import
import tensorflow as tf
from tensorflow.python.ops import variable_scope

# module import
from .utils import get_product_scores
from .model_utils import UQP_nce_loss
from .BasicModel import BasicModel

class HEM(BasicModel):
    def __init__(self, dataset, hparams_setting, forward_only):
        """
        Args:
            model_settings: (dictionary): a set of key, value pairs for model hyperparams tunning.
        """
        # dataset parameters
        self.dataset = dataset
        self.vocab_size = self.dataset.vocab_size
        self.user_size = self.dataset.user_size
        self.product_size = self.dataset.product_size
        self.query_max_length = self.dataset.query_max_length
        self.vocab_distribute = self.dataset.vocab_distribute
        self.review_distribute = self.dataset.review_distribute
        self.product_distribute = self.dataset.product_distribute

        # model hparams
        self.hparams = tf.contrib.training.HParams(
         window_size=5,
         embed_size=300,
         max_gradient_norm=5.0,
         learning_rate=0.5,
         L2_lambda=0.0,
         query_weight=0.5,
         negative_sample=5
        )
        self.hparams.override_from_dict(hparams_setting)

        # model params settings
        self.negative_sample = self.hparams.negative_sample
        self.embed_size = self.hparams.embed_size
        self.window_size = self.hparams.window_size
        self.max_gradient_norm = self.hparams.max_gradient_norm
        self.init_learning_rate = self.hparams.learning_rate
        self.L2_lambda = self.hparams.L2_lambda
        #self.similarity_func = similarity_func
        #self.similarity_func = similarity_func
        self.global_step = tf.Variable(0, trainable=False)
        if self.hparams.query_weight >= 0:
            self.Wu = tf.Variable(self.hparams.query_weight, name="user_weight", dtype=tf.float32, trainable=False)
        else:
            self.Wu = tf.sigmoid(tf.Variable(0, name="user_weight", dtype=tf.float32))

        # create placeholder
        self._create_placeholder()

        # Setup model
        print('L2 lambda ' + str(self.hparams.L2_lambda))

        # Training losses
        self.loss = None
        self.loss = self._build_embedding_graph_and_get_loss()

        # Gradients and SGD update operation for training the model.
        if not forward_only:
            self.updates = self._build_optimizer_and_get_updates()
        else:
            self.product_scores = get_product_scores(self, self.user_idxs, self.query_word_idxs)

        tf.summary.scalar('Learning Rate', self.learning_rate, collections=['train'])
        tf.summary.scalar('Loss', self.loss, collections=['train'])

        self.train_summary = tf.summary.merge_all(key='train')

        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, input_feed, forward_only, test_mode='product_scores'):
        if not forward_only:
            output_feed = [self.updates,    # Update Op that does SGD.
                            self.loss,      # Loss for this batch.
                            self.train_summary # Summarization statistics
                          ]
        else:
            if test_mode == 'output_embedding':
                output_feed = [self.user_emb, self.product_emb, self.Wu, self.word_emb, self.word_bias]
            else:
                output_feed = [self.product_scores] #negative instance output

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs    # loss, no outputs, Gradient norm.
        else:
            if test_mode == 'output_embedding':
                return outputs[:4], outputs[4:]
            else:
                return outputs[0], None    # product scores to input user

    def _create_placeholder(self):
        # Feeds for inputs
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        self.review_idxs = tf.placeholder(tf.int64, shape=[None], name="review_idxs")
        self.user_idxs = tf.placeholder(tf.int64, shape=[None], name="user_idxs")
        self.product_idxs = tf.placeholder(tf.int64, shape=[None], name="product_idxs")
        self.word_idxs = tf.placeholder(tf.int64, shape=[None], name="word_idxs")
        self.query_word_idxs = tf.placeholder(tf.int64, shape=[None, self.query_max_length], name="query_word_idxs")

    def _build_embedding_graph_and_get_loss(self, scope = None):
        with variable_scope.variable_scope(scope or "embedding_graph"):
            # Word embeddings.
            init_width = 0.5 / self.embed_size
            self.word_emb = tf.Variable( tf.random_uniform(
                                [self.vocab_size+1, self.embed_size], -init_width, init_width),
                                name="word_emb")
            #self.word_emb = tf.concat(axis=0,values=[self.word_emb,self.PAD_embed])
            self.word_bias = tf.Variable(tf.zeros([self.vocab_size+1]), name="word_b")

            # user/product embeddings.
            self.user_emb =    tf.Variable( tf.zeros([self.user_size, self.embed_size]),
                                name="user_emb")
            self.user_bias =    tf.Variable( tf.zeros([self.user_size]), name="user_b")
            self.product_emb =    tf.Variable( tf.zeros([self.product_size, self.embed_size]),
                                name="product_emb")
            self.product_bias =    tf.Variable( tf.zeros([self.product_size]), name="product_b")

            return UQP_nce_loss(self, self.user_idxs, self.query_word_idxs, self.product_idxs, self.review_idxs,
                                            self.word_idxs)

    def _build_optimizer_and_get_updates(self):
        params = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.gradients = tf.gradients(self.loss, params)

        self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
                                                                 self.max_gradient_norm)
        self.updates = opt.apply_gradients(zip(self.clipped_gradients, params),
                                         global_step=self.global_step)
        return self.updates
