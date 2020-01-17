from __future__ import print_function

import random

import numpy as np

from esrt.engine.base_input_feed import BaseInputFeed


class ZAMInputFeed(BaseInputFeed):
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def setup_data_set(self, data_set, words_to_train):
        self.data_set = data_set
        self.words_to_train = words_to_train
        self.finished_word_num = 0
        if self.model.net_struct == 'hdc':
            self.model.need_context = True

    def intialize_epoch(self, training_seq):
        self.train_seq = training_seq
        self.review_size = len(self.train_seq)
        self.cur_review_i = 0
        self.cur_word_i = 0

    def get_train_batch(self, debug=False):
        user_idxs, product_idxs, review_idxs, word_idxs, context_word_idxs = [],[],[],[],[]
        query_word_idxs = []
        history_product_idxs = []
        input_history_length = []
        learning_rate = self.model.init_learning_rate * max(0.0001,
                                    1.0 - self.finished_word_num / self.words_to_train)
        review_idx = self.train_seq[self.cur_review_i]
        user_idx = self.data_set.review_info[review_idx][0]
        product_idx = self.data_set.review_info[review_idx][1]
        query_idx = random.choice(self.data_set.product_query_idx[product_idx])
        text_list = self.data_set.review_text[review_idx]
        text_length = len(text_list)
        while len(word_idxs) < self.batch_size:
            #print('review %d word %d word_idx %d' % (review_idx, self.cur_word_i, text_list[self.cur_word_i]))
            #if sample this word
            if self.data_set.sub_sampling_rate == None or random.random() < self.data_set.sub_sampling_rate[text_list[self.cur_word_i]]:
                user_idxs.append(user_idx)
                product_idxs.append(product_idx)
                product_history_idxs, history_length = self.model.dataset.get_history_products(user_idx, product_idx, self.model.max_history_length)
                history_product_idxs.append(product_history_idxs)
                input_history_length.append(history_length)
                query_word_idxs.append(self.data_set.query_words[query_idx])
                review_idxs.append(review_idx)
                word_idxs.append(text_list[self.cur_word_i])


            #move to the next
            self.cur_word_i += 1
            self.finished_word_num += 1
            if self.cur_word_i == text_length:
                self.cur_review_i += 1
                if self.cur_review_i == self.review_size:
                    break
                self.cur_word_i = 0
                review_idx = self.train_seq[self.cur_review_i]
                user_idx = self.data_set.review_info[review_idx][0]
                product_idx = self.data_set.review_info[review_idx][1]
                query_idx = random.choice(self.data_set.product_query_idx[product_idx])
                text_list = self.data_set.review_text[review_idx]
                text_length = len(text_list)



        has_next = False if self.cur_review_i == self.review_size else True

        # create input feed
        input_feed = {}
        input_feed[self.model.learning_rate.name] = learning_rate
        input_feed[self.model.user_idxs.name] = user_idxs
        input_feed[self.model.product_idxs.name] = product_idxs
        input_feed[self.model.history_product_idxs] = history_product_idxs
        input_feed[self.model.input_history_length.name] = input_history_length
        input_feed[self.model.query_word_idxs.name] = query_word_idxs
        input_feed[self.model.review_idxs.name] = review_idxs
        input_feed[self.model.word_idxs.name] = word_idxs


        if debug:
            for i in range(len(product_idxs)):
                print("user idx: ", user_idxs[i])
                print("product idx: ", product_idxs[i])
                print("history products: ", history_product_idxs[i])
                print("len is: ", len(history_product_idxs[i]))
                print("=================")
            print("Finish a batch")
        return input_feed, has_next

    def prepare_test_epoch(self,debug=False):
        self.test_user_query_set = set()
        self.test_seq = []
        for review_idx in range(len(self.data_set.review_info)):
            user_idx = self.data_set.review_info[review_idx][0]
            product_idx = self.data_set.review_info[review_idx][1]
            for query_idx in self.data_set.product_query_idx[product_idx]:
                if (user_idx, query_idx) not in self.test_user_query_set:
                    self.test_user_query_set.add((user_idx, query_idx))
                    self.test_seq.append((user_idx, product_idx, query_idx, review_idx))
        self.cur_uqr_i = 0
        if debug:
        # only print first 10 lines
            line = 0
            while line < 10:
                print("test seq element: ")
                print(self.test_seq[line])
                print(self.data_set.review_info[line])
                line+=1
            print("its length is: ", len(self.test_seq))


    def get_test_batch(self, debug=False):
        user_idxs, product_idxs, review_idxs, word_idxs, context_word_idxs = [],[],[],[],[]
        query_word_idxs = []
        history_product_idxs = []
        input_history_length = []
        learning_rate = self.model.init_learning_rate * max(0.0001,
                                    1.0 - self.finished_word_num / self.words_to_train)
        start_i = self.cur_uqr_i
        user_idx, product_idx, query_idx, review_idx = self.test_seq[self.cur_uqr_i]

        while len(user_idxs) < self.batch_size:
            text_list = self.data_set.review_text[review_idx]
            user_idxs.append(user_idx)
            product_idxs.append(product_idx)
            product_history_idxs, history_length = self.model.dataset.get_history_products(user_idx, product_idx, self.model.max_history_length)
            history_product_idxs.append(product_history_idxs)
            input_history_length.append(history_length)
            query_word_idxs.append(self.data_set.query_words[query_idx])
            review_idxs.append(review_idx)
            word_idxs.append(text_list[0])

            #move to the next review
            self.cur_uqr_i += 1
            if self.cur_uqr_i == len(self.test_seq):
                break
            user_idx, product_idx, query_idx, review_idx = self.test_seq[self.cur_uqr_i]

        has_next = False if self.cur_uqr_i == len(self.test_seq) else True
        # create input feed
        input_feed = {}
        input_feed[self.model.learning_rate.name] = learning_rate
        input_feed[self.model.user_idxs.name] = user_idxs
        input_feed[self.model.product_idxs.name] = product_idxs
        input_feed[self.model.history_product_idxs.name] = history_product_idxs
        input_feed[self.model.input_history_length.name] = input_history_length
        input_feed[self.model.query_word_idxs.name] = query_word_idxs
        input_feed[self.model.review_idxs.name] = review_idxs
        input_feed[self.model.word_idxs.name] = word_idxs

        if debug:
            print("This is a test batch")
        return input_feed, has_next, self.test_seq[start_i:self.cur_uqr_i]
