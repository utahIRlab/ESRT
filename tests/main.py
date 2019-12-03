"""Training and testing the hierarchical embedding model for personalized product search

See the following paper for more information on the hierarchical embedding model.

    * Qingyao Ai, Yongfeng Zhang, Keping Bi, Xu Chen, W. Bruce Croft. 2017. Learning a Hierarchical Embedding Model for Personalized ProductSearch. In Proceedings of SIGIR '17
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange    # pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf
import yaml


from product_embedding_zoo import utils
from product_embedding_zoo import input_feed


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.90,
                            "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                            "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("subsampling_rate", 1e-4,
                            "The rate to subsampling.")
tf.app.flags.DEFINE_float("L2_lambda", 0.0,
                            "The lambda for L2 regularization.")
tf.app.flags.DEFINE_float("query_weight", 0.5,
                            "The weight for query.")
tf.app.flags.DEFINE_float("dynamic_weight", 0.5, "The weight for the dynamic relationship [0.0, 1.0].")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
#rank list size should be read from data
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("input_train_dir", "", "The directory of training and testing data")
tf.app.flags.DEFINE_string("train_dir", "./tmp/", "Model directory & output directory")
tf.app.flags.DEFINE_string("logging_dir", ".log/", "Log directory")
tf.app.flags.DEFINE_string("similarity_func", "product", "Select similarity function, which could be product, cosine and bias_product")
tf.app.flags.DEFINE_string("net_struct", "simplified_fs", "Specify network structure parameters. Please read readme.txt for details.")
tf.app.flags.DEFINE_integer("embed_size", 100, "Size of each embedding.")
tf.app.flags.DEFINE_integer("window_size", 5, "Size of context window.")
tf.app.flags.DEFINE_integer("max_train_epoch", 5,
                            "Limit on the epochs of training (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("seconds_per_checkpoint", 3600,
                            "How many seconds to wait before storing embeddings.")
tf.app.flags.DEFINE_integer("negative_sample", 5,
                            "How many samples to generate for negative sampling.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for testing.")
tf.app.flags.DEFINE_string("test_mode", "product_scores", "Test modes: product_scores -> output ranking results and ranking scores; output_embedding -> output embedding representations for users, items and words. (default is product_scores)")
tf.app.flags.DEFINE_integer("rank_cutoff", 100,
                            "Rank cutoff for output ranklists.")

tf.app.flags.DEFINE_string("setting_file", "./example/exp1.yaml", "a yaml contains all model settings.")


FLAGS = tf.app.flags.FLAGS

HPARAMS_DICT = {
    "window_size":FLAGS.window_size,
    "embed_size":FLAGS.embed_size,
    "max_gradient_norm":FLAGS.max_gradient_norm,
    "init_learning_rate": FLAGS.learning_rate,
    "L2_lambda": FLAGS.L2_lambda,
    "query_weight":FLAGS.query_weight,
    "dynamic_weight": FLAGS.dynamic_weight,
    "net_struct": FLAGS.net_struct,
    "similarity_func": FLAGS.similarity_func,
    "negative_sample": FLAGS.negative_sample
}


def create_model(session, model_name, forward_only, data_set):
    """Create translation model and initialize or load parameters in session."""
    print("Create a learning model %s"%model_name)
    model = utils.find_class(model_name)(data_set, HPARAMS_DICT, forward_only)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt:
        ckpt_file = FLAGS.train_dir + ckpt.model_checkpoint_path.split('/')[-1]
        #print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        #model.saver.restore(session, ckpt.model_checkpoint_path)
        print("Reading model parameters from %s" % ckpt_file)
        model.saver.restore(session, ckpt_file)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def train(exp_settings):
    # Hack the file path  when use python -m test.main
    data_dir = os.path.join(os.path.dirname(__file__), '..', FLAGS.data_dir)
    input_train_dir = os.path.join(os.path.dirname(__file__), '..', FLAGS.input_train_dir)
    # Prepare data.
    print("Reading data in %s" % data_dir)
    print("------experiment settings' key, value pairs: ----")
    for key, val in exp_settings.items():
        print(key, val)
    dataset_str = exp_settings['arch']['dataset_type']
    input_feed_str = exp_settings['arch']['input_feed']
    model_str = exp_settings['arch']['learning_algorithm']

    data_set = utils.find_class(dataset_str)(data_dir, input_train_dir, 'train')
    data_set.sub_sampling(FLAGS.subsampling_rate)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.log_device_placement=True
    with tf.Session(config=config) as sess:
        # Create model.
        print("Creating model")
        model = create_model(sess,model_str,False, data_set)
        print("Create a input feed module %s"%input_feed_str)
        input_feed = utils.find_class(input_feed_str)(model, FLAGS.batch_size)
        compat_input_feed = CompatInputFeed(input_feed)

        train_writer = tf.summary.FileWriter(FLAGS.logging_dir, sess.graph)
        print('Start training')
        words_to_train = float(FLAGS.max_train_epoch * data_set.word_count) + 1
        previous_words = 0.0
        start_time = time.time()
        step_time, loss = 0.0, 0.0
        current_epoch = 0
        current_step = 0
        get_batch_time = 0.0
        training_seq = [i for i in xrange(data_set.review_size)]
        input_feed.setup_data_set(data_set, words_to_train)
        while True:
            random.shuffle(training_seq)
            input_feed.intialize_epoch(training_seq)
            has_next = True
            while has_next:
                time_flag = time.time()
                batch_input_feed, has_next = input_feed.get_train_batch()
                get_batch_time += time.time() - time_flag

                # output params
                #word_idxs = batch_input_feed[model.word_idxs.name]
                #learning_rate = batch_input_feed[model.learning_rate.name]
                word_idxs = compat_input_feed.word_idxs(batch_input_feed, model)
                learning_rate = compat_input_feed.learning_rate(batch_input_feed, model)

                if len(word_idxs) > 0:
                    time_flag = time.time()
                    step_loss, summary = model.step(sess, batch_input_feed, False, file_writer=train_writer)
                    #train_writer.add_run_metadata(run_metadata, global_step=self.global_stepi)
                    #print("The summaries are: ", summary)
                    train_writer.add_summary(summary, model.global_step.eval())
                    #step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                    loss += step_loss / FLAGS.steps_per_checkpoint
                    current_step += 1
                    step_time += time.time() - time_flag

                # Once in a while, we print statistics.
                if current_step % FLAGS.steps_per_checkpoint == 0:
                    print("Epoch %d Words %d/%d: lr = %5.3f loss = %6.2f words/sec = %5.2f prepare_time %.2f step_time %.2f\r" %
                            (current_epoch, input_feed.finished_word_num, input_feed.words_to_train, learning_rate, loss,
                                (input_feed.finished_word_num- previous_words)/(time.time() - start_time), get_batch_time, step_time), end="")
                    step_time, loss = 0.0, 0.0
                    current_step = 1
                    get_batch_time = 0.0
                    sys.stdout.flush()
                    previous_words = input_feed.finished_word_num
                    start_time = time.time()
                    #print('time: ' + str(time.time() - last_check_point_time))
                    #if time.time() - last_check_point_time > FLAGS.seconds_per_checkpoint:
                    #    checkpoint_path_best = os.path.join(FLAGS.train_dir, "ProductSearchEmbedding.ckpt")
                    #    model.saver.save(sess, checkpoint_path_best, global_step=model.global_step)

            current_epoch += 1
            #checkpoint_path_best = os.path.join(FLAGS.train_dir, "ProductSearchEmbedding.ckpt")
            #model.saver.save(sess, checkpoint_path_best, global_step=model.global_step)
            if current_epoch >= FLAGS.max_train_epoch:
                break
        checkpoint_path_best = os.path.join(FLAGS.train_dir, "ProductSearchEmbedding.ckpt")
        #logging.INFO("The checkpoint best path is in:  %s"%(checkpoint_path_best))
        model.saver.save(sess, checkpoint_path_best, global_step=model.global_step)


def get_product_scores(exp_settings):
    # Prepare data.
    # Hack the file path  when use python -m test.main
    data_dir = os.path.join(os.path.dirname(__file__), '..', FLAGS.data_dir)
    input_train_dir = os.path.join(os.path.dirname(__file__), '..', FLAGS.input_train_dir)
    print("Reading data in %s" % data_dir)
    dataset_str = exp_settings['arch']['dataset_type']
    input_feed_str = exp_settings['arch']['input_feed']

    data_set = utils.find_class(dataset_str)(data_dir, input_train_dir, 'test')
    data_set.read_train_product_ids(input_train_dir)
    current_step = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Create model.
        print("Read model")
        model = create_model(sess,  exp_settings['arch']['learning_algorithm'], True, data_set)
        input_feed= utils.find_class(input_feed_str)(model, FLAGS.batch_size)
        user_ranklist_map = {}
        user_ranklist_score_map = {}
        print('Start Testing')
        words_to_train = float(FLAGS.max_train_epoch * data_set.word_count) + 1
        test_seq = [i for i in xrange(data_set.review_size)]
        input_feed.setup_data_set(data_set, words_to_train)
        input_feed.intialize_epoch(test_seq)
        input_feed.prepare_test_epoch()
        has_next = True
        while has_next:
            batch_input_feed, has_next, uqr_pairs = input_feed.get_test_batch()

            # get params
            user_idxs = batch_input_feed[model.user_idxs.name]
            if len(user_idxs) > 0:
                user_product_scores, _ = model.step(sess, batch_input_feed, True)
                current_step += 1

            # record the results
            for i in xrange(len(uqr_pairs)):
                u_idx, p_idx, q_idx, r_idx = uqr_pairs[i]
                sorted_product_idxs = sorted(range(len(user_product_scores[i])),
                                    key=lambda k: user_product_scores[i][k], reverse=True)
                user_ranklist_map[(u_idx, q_idx)],user_ranklist_score_map[(u_idx, q_idx)] = data_set.compute_test_product_ranklist(u_idx,
                                                user_product_scores[i], sorted_product_idxs, FLAGS.rank_cutoff) #(product name, rank)
            if current_step % FLAGS.steps_per_checkpoint == 0:
                print("Finish test review %d/%d\r" %
                        (input_feed.cur_uqr_i, len(input_feed.test_seq)), end="")

    data_set.output_ranklist(user_ranklist_map, user_ranklist_score_map, FLAGS.train_dir, FLAGS.similarity_func)
    return

def output_embedding(exp_settings):
    # Prepare data.
    # Hack the file path  when use python -m test.main
    data_dir = os.path.join(os.path.dirname(__file__), '..', FLAGS.data_dir)
    input_train_dir = os.path.join(os.path.dirname(__file__), '..', FLAGS.input_train_dir)
    print("Reading data in %s" % data_dir)
    dataset_str = exp_settings['arch']['dataset_type']
    input_feed_str = exp_settings['arch']['input_feed']

    data_set = utils.find_class(dataset_str)(data_dir, input_train_dir, 'test')
    data_set.read_train_product_ids(FLAGS.input_train_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Create model.
        print("Read model")
        model = create_model(sess,  exp_settings['arch']['learning_algorithm'], True, data_set)
        input_feed = utils.find_class(input_feed_str)(model, FLAGS.batch_size)
        user_ranklist_map = {}
        print('Start Testing')
        words_to_train = float(FLAGS.max_train_epoch * data_set.word_count) + 1
        test_seq = [i for i in xrange(data_set.review_size)]
        input_feed.setup_data_set(data_set, words_to_train)
        input_feed.intialize_epoch(test_seq)
        input_feed.prepare_test_epoch()
        has_next = True
        user_idxs, product_idxs, query_word_idxs, review_idxs, word_idxs, context_word_idxs, learning_rate, has_next, uqr_pairs = input_feed.get_test_batch()

        if len(user_idxs) > 0:
            part_1 , part_2 = model.step(sess, learning_rate, user_idxs, product_idxs, query_word_idxs,
                        review_idxs, word_idxs, context_word_idxs, True, FLAGS.test_mode)

            # record the results
            user_emb = part_1[0]
            product_emb = part_1[1]
            Wu = part_1[2]
            data_set.output_embedding(user_emb, FLAGS.train_dir + 'user_emb.txt')
            data_set.output_embedding(product_emb, FLAGS.train_dir + 'product_emb.txt')
            data_set.output_embedding(Wu, FLAGS.train_dir + 'Wu.txt')
    return

def _parse_exp_settings(settings_file):
    with open(settings_file, 'r') as sf:
        exp_settings = yaml.load(sf, Loader=yaml.FullLoader)
    return exp_settings

def main(_):
    exp_settings = _parse_exp_settings(FLAGS.setting_file)
    if FLAGS.input_train_dir == "":
        FLAGS.input_train_dir = FLAGS.data_dir

    if FLAGS.decode:
        if FLAGS.test_mode == 'output_embedding':
            output_embedding(exp_settings)
        else:
            get_product_scores(exp_settings)
    else:
        train(exp_settings)
##################################
###### Helper utils ##############
##################################
class CompatInputFeed():
    """make different input_feed(HEMInputFeed, DREMInputFeed,...) object compatibility,
    when they use their attributes.
    """
    def __init__(self, input_feed):
        self.input_feed = input_feed

    def word_idxs(self,batch_input_feed, model) :
        if isinstance(self.input_feed, input_feed.HEMInputFeed):
            return batch_input_feed[model.word_idxs.name]
        elif isinstance(self.input_feed, input_feed.DREMInputFeed):
            #print("the current input feed is %s"%str(input_feed.DREMInputFeed))
            return batch_input_feed[model.relation_dict['word']['idxs'].name]
        else:
            raise ValueError("The input feed class %s is not defined"%str(self.input_feed))
    def learning_rate(self,batch_input_feed, model):
        if isinstance(self.input_feed, input_feed.HEMInputFeed):
            return batch_input_feed[model.learning_rate.name]
        elif isinstance(self.input_feed, input_feed.DREMInputFeed):
            return batch_input_feed[model.learning_rate.name]
        else:
            raise ValueError("The input feed class %s is not defined"%str(self.input_feed))

if __name__ == "__main__":
    print(__file__)
    tf.app.run()
