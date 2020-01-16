from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
from .BasicInputFeed import BasicInputFeed


class HEMInputFeed(BasicInputFeed):
    """
    The HEM input feed layer implemention for product search embedding
    """
    def __init__(self, model, data_set, exp_settings ,session):
        self.model = model
        # the set_up_data_set part in original code
        # TODO [ ]: modify FLGAS
        self.data_set = data_set
        self.words_to_train = float(exp_settings['max_train_epoch'] * data_set.word_count) + 1
        self.finished_word_num = 0

        # the initialize_epoch part in original code
        self.train_seq = [i for i in range(self.data_set.review_size)]
        self.record_size = len(self.train_seq)
        self.cur_review_i = 0
        self.cur_word_i = 0

    def shuffle(self):
        random.shuffle(self.train_seq)

    def initialize_epoch(self):
        self.cur_review_i = 0
        self.cur_word_i = 0

    def get_train_batch(self, batch_size):
        user_idxs, product_idxs, review_idxs, word_idxs = [], [], [], []
        query_word_idxs = []
        learning_rate = self.model.init_learning_rate*max(0.0001, 1.0 - self.finished_word_num / self.words_to_train)

        review_idx = self.train_seq[self.cur_review_i]
        user_idx = self.data_set.review_info[review_idx][0]
        product_idx = self.data_set.review_info[review_idx][1]
        query_idx = random.choice(self.data_set.product_query_idx[product_idx])
        text_list = self.data_set.review_text[review_idx]
        text_length = len(text_list)


        while len(word_idxs) < batch_size:
            if self.data_set.sub_sampling_rate is None or random.random() < self.data_set.sub_sampling_rate[text_list[self.cur_word_i]]:
                user_idxs.append(user_idx)
                product_idxs.append(product_idx)
                query_word_idxs.append(self.data_set.query_words[query_idx])
                review_idxs.append(review_idx)
                word_idxs.append(text_list[self.cur_word_i])

            #move to the next
            self.cur_word_i += 1
            self.finished_word_num += 1
            if self.cur_word_i == text_length:
                self.cur_review_i += 1
                if self.cur_review_i == self.data_set.review_size:
                    break
                self.cur_word_i = 0
                review_idx = self.train_seq[self.cur_review_i]
                user_idx = self.data_set.review_info[review_idx][0]
                product_idx = self.data_set.review_info[review_idx][1]
                query_idx = random.choice(self.data_set.product_query_idx[product_idx])
                text_list = self.data_set.review_text[review_idx]
                text_length = len(text_list)


        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        input_feed[self.model.learning_rate.name] = learning_rate
        input_feed[self.model.user_idxs.name] = user_idxs  # dim: (batch_size, 1)
        input_feed[self.model.product_idxs.name] = product_idxs # dim: (batch_size, 1)
        input_feed[self.model.query_word_idxs.name] = query_word_idxs  # dim: (batch_size, query_max_length)
        input_feed[self.model.review_idxs.name] = review_idxs  # dim: (batch_size,  1)
        input_feed[self.model.word_idxs.name] = word_idxs  # dim: (batch_size, 1)


        has_next = False if self.cur_review_i == self.data_set.review_size else True
        return input_feed, has_next
