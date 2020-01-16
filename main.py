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
import tensorflow.compat.v1 as tf
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from esrt import input_feed, utils
from esrt.engine.param_table import ParamTable



tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for testing.")
tf.app.flags.DEFINE_string("test_mode", "product_scores", "Test modes: product_scores -> output ranking results and ranking scores; output_embedding -> output embedding representations for users, items and words. (default is product_scores)")
#tf.app.flags.DEFINE_integer("rank_cutoff", 100,
#                            "Rank cutoff for output ranklists.")

tf.app.flags.DEFINE_string("setting_file", "./example/exp1.yaml", "a yaml contains all model settings.")


FLAGS = tf.app.flags.FLAGS

def create_model(session, model_name, hparams, forward_only, data_set, model_dir):
    """Create translation model and initialize or load parameters in session."""
    print("Create a learning model %s"%model_name)
    model = utils.find_class(model_name)(data_set, hparams, forward_only)
    model.build()
    print("reading ckpt file from ", model_dir)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt:
        ckpt_file = model_dir + ckpt.model_checkpoint_path.split('/')[-1]
        #print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        #model.saver.restore(session, ckpt.model_checkpoint_path)
        print("Reading model parameters from %s" % ckpt_file)
        model.saver.restore(session, ckpt_file)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def train():
    # parse exp settings file
    aparams, dparams, eparams, hparams = _parse_exp_settings(FLAGS.setting_file)

    data_dir = dparams['data_dir']
    input_train_dir = dparams['input_train_dir']

    # Prepare data.
    print("Reading data in %s" % data_dir)

    # get module(arch) name  information
    dataset_str = aparams['dataset_type']
    input_feed_str = aparams['input_feed']
    model_str = aparams['learning_algorithm']

    data_set = utils.find_class(dataset_str)(data_dir, input_train_dir, 'train')
    data_set.sub_sampling(eparams['subsampling_rate'])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.log_device_placement=True
    with tf.Session(config=config) as sess:
        # Create model.
        print("Creating model")
        model = create_model(sess, model_str, hparams, False, data_set, dparams['model_dir'])
        print("Create a input feed module %s"%input_feed_str)
        input_feed = utils.find_class(input_feed_str)(model, hparams['batch_size'])
        compat_input_feed = CompatInputFeed(input_feed)

        print('Start training')
        words_to_train = float(eparams['max_train_epoch'] * data_set.word_count) + 1
        previous_words = 0.0
        start_time = time.time()
        step_time, loss = 0.0, 0.0
        current_epoch = 0
        current_step = 0
        get_batch_time = 0.0
        training_seq = [i for i in range(data_set.review_size)]
        input_feed.setup_data_set(data_set, words_to_train)
        while True:
            random.shuffle(training_seq)
            input_feed.intialize_epoch(training_seq)
            has_next = True
            while has_next:
                time_flag = time.time()
                batch_input_feed, has_next = input_feed.get_train_batch(debug=False)
                get_batch_time += time.time() - time_flag

                word_idxs = compat_input_feed.word_idxs(batch_input_feed, model)
                learning_rate = compat_input_feed.learning_rate(batch_input_feed, model)

                if len(word_idxs) > 0:
                    time_flag = time.time()
                    step_loss = model.step(sess, batch_input_feed, False)
                    #step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                    loss += step_loss / eparams['steps_per_checkpoint']
                    current_step += 1
                    #print(step_loss)
                    step_time += time.time() - time_flag

                # Once in a while, we print statistics.
                if current_step % eparams['steps_per_checkpoint'] == 0:
                    print("Epoch %d Words %d/%d: lr = %5.3f loss = %6.2f words/sec = %5.2f prepare_time %.2f step_time %.2f\r" %
                            (current_epoch, input_feed.finished_word_num, input_feed.words_to_train, learning_rate, loss,
                                (input_feed.finished_word_num- previous_words)/(time.time() - start_time), get_batch_time, step_time), end="")
                    step_time, loss = 0.0, 0.0
                    current_step = 1
                    get_batch_time = 0.0
                    sys.stdout.flush()
                    previous_words = input_feed.finished_word_num
                    start_time = time.time()

            current_epoch += 1
            if not os.path.exists(dparams['model_dir']):
                os.mkdir(dparams['model_dir'])
            checkpoint_path_best = os.path.join(dparams['model_dir'], "ProductSearchEmbedding.ckpt")
            model.saver.save(sess, checkpoint_path_best, global_step=model.global_step)
            if current_epoch >= eparams['max_train_epoch']:
                break
        checkpoint_path_best = os.path.join(dparams['model_dir'], "ProductSearchEmbedding.ckpt")
        model.saver.save(sess, checkpoint_path_best, global_step=model.global_step)


def get_product_scores():
    # parse exp settings file
    aparams, dparams, eparams, hparams = _parse_exp_settings(FLAGS.setting_file)

    data_dir = dparams['data_dir']
    input_train_dir = dparams['input_train_dir']

    # read data
    print("Reading data in %s" % data_dir)

    # get module(arch) name  information
    dataset_str = aparams['dataset_type']
    input_feed_str = aparams['input_feed']
    model_str = aparams['learning_algorithm']

    # create dataset object
    data_set = utils.find_class(dataset_str)(data_dir, input_train_dir, 'test')
    data_set.read_train_product_ids(input_train_dir)
    current_step = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Create model.
        print("Read model")
        model = create_model(sess,  model_str, hparams, True, data_set, dparams['model_dir'])
        input_feed= utils.find_class(input_feed_str)(model, hparams['batch_size'])
        user_ranklist_map = {}
        user_ranklist_score_map = {}
        print('Start Testing')
        words_to_train = float(eparams['max_train_epoch'] * data_set.word_count) + 1
        test_seq = [i for i in range(data_set.review_size)]
        input_feed.setup_data_set(data_set, words_to_train)
        input_feed.intialize_epoch(test_seq)
        input_feed.prepare_test_epoch(debug=True)
        has_next = True
        while has_next:
            batch_input_feed, has_next, uqr_pairs = input_feed.get_test_batch(debug=True)

            # get params
            user_idxs = batch_input_feed[model.user_idxs.name]
            if len(user_idxs) > 0:
                user_product_scores, _ = model.step(sess, batch_input_feed, True)
                current_step += 1
            print("product scores: ")
            for uidx in range(len(user_product_scores)):
                if uidx > 10:
                    break

            # record the results
            for i in range(len(uqr_pairs)):
                u_idx, p_idx, q_idx, r_idx = uqr_pairs[i]
                sorted_product_idxs = sorted(range(len(user_product_scores[i])),
                                    key=lambda k: user_product_scores[i][k], reverse=True)
                user_ranklist_map[(u_idx, q_idx)],user_ranklist_score_map[(u_idx, q_idx)] = data_set.compute_test_product_ranklist(u_idx,
                                                user_product_scores[i], sorted_product_idxs, eparams['rank_cutoff']) #(product name, rank)
            if current_step % eparams['steps_per_checkpoint']== 0:
                print("Finish test review %d/%d\r" %
                        (input_feed.cur_uqr_i, len(input_feed.test_seq)), end="")

    data_set.output_ranklist(user_ranklist_map, user_ranklist_score_map, dparams['model_dir'], hparams['similarity_func'], debug=True)
    return

def output_embedding(exp_settings):
     # parse exp settings file
    aparams, dparams, eparams, hparams = _parse_exp_settings(FLAGS.setting_file)

    # Hack the file path  when use python -m test.main
    data_dir = os.path.join(os.path.dirname(__file__), '..', eparams.data_dir)
    input_train_dir = os.path.join(os.path.dirname(__file__), '..', eparams.input_train_dir)

    # read data
    print("Reading data in %s" % data_dir)

    # get module(arch) name  information
    dataset_str = aparams.dataset_type
    input_feed_str = aparams.input_feed
    model_str = aparams.learning_algorithm

    # create dataset object
    data_set = utils.find_class(dataset_str)(data_dir, input_train_dir, 'test')
    data_set.read_train_product_ids(dparams.input_train_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Create model.
        print("Read model")
        model = create_model(sess,  exp_settings['arch']['learning_algorithm'], hparams, True, data_set, dparams.model_dir)
        input_feed = utils.find_class(input_feed_str)(model, hparams.batch_size)
        user_ranklist_map = {}
        print('Start Testing')
        words_to_train = float(eparams.max_train_epoch * data_set.word_count) + 1
        test_seq = [i for i in range(data_set.review_size)]
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
            data_set.output_embedding(user_emb, dparams.model_dir + 'user_emb.txt')
            data_set.output_embedding(product_emb, dparams.model_dir + 'product_emb.txt')
            data_set.output_embedding(Wu, dparams.model_dir + 'Wu.txt')
    return

def _parse_exp_settings(settings_file):
    hparams = ParamTable()
    hparams.update_from_yaml(settings_file)

    with open(settings_file, 'r') as f:
        tdict = yaml.load(f, Loader=yaml.SafeLoader)
        aparams = tdict['arch']
        dparams = tdict['data']
        eparams = tdict['experiment']

    return aparams, dparams, eparams, hparams

def main(_):
    exp_settings = _parse_exp_settings(FLAGS.setting_file)
    #if FLAGS.input_train_dir == "":
        #FLAGS.input_train_dir = FLAGS.data_dir

    if FLAGS.decode:
        if FLAGS.test_mode == 'output_embedding':
            output_embedding()
        else:
            get_product_scores()
    else:
        train()
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
        elif isinstance(self.input_feed, input_feed.AEMInputFeed):
            return batch_input_feed[model.word_idxs.name]
        elif isinstance(self.input_feed, input_feed.DREMInputFeed):
            #print("the current input feed is %s"%str(input_feed.DREMInputFeed))
            return batch_input_feed[model.relation_dict['word']['idxs'].name]
        else:
            raise ValueError("The input feed class %s is not defined"%str(self.input_feed))
    def learning_rate(self,batch_input_feed, model):
        if isinstance(self.input_feed, input_feed.HEMInputFeed):
            return batch_input_feed[model.learning_rate.name]
        if isinstance(self.input_feed, input_feed.AEMInputFeed):
            return batch_input_feed[model.learning_rate.name]
        elif isinstance(self.input_feed, input_feed.DREMInputFeed):
            return batch_input_feed[model.learning_rate.name]
        else:
            raise ValueError("The input feed class %s is not defined"%str(self.input_feed))

if __name__ == "__main__":
    print(__file__)
    tf.app.run()
