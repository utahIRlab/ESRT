import random

# 3rd import
import numpy as np
from six.moves import xrange


class HEMInputFeed(object):
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

    def get_train_batch(self,debug=False):
        user_idxs, product_idxs, review_idxs, word_idxs, context_word_idxs = [],[],[],[],[]
        query_word_idxs = []
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
                query_word_idxs.append(self.data_set.query_words[query_idx])
                review_idxs.append(review_idx)
                word_idxs.append(text_list[self.cur_word_i])
                if self.model.need_context:
                    i = self.cur_word_i
                    start_index = i - self.model.window_size if i - self.model.window_size > 0 else 0
                    context_word_list = text_list[start_index:i] + text_list[i+1:i+self.model.window_size+1]
                    while len(context_word_list) < 2 * self.model.window_size:
                        context_word_list += text_list[start_index:start_index+2*self.model.window_size-len(context_word_list)]
                    context_word_idxs.append(context_word_list)

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

        batch_context_word_idxs = None
        length = len(word_idxs)
        if self.model.need_context:
            batch_context_word_idxs = []
            for length_idx in xrange(2 * self.model.window_size):
                batch_context_word_idxs.append(np.array([context_word_idxs[batch_idx][length_idx]
                        for batch_idx in xrange(length)], dtype=np.int64))

        has_next = False if self.cur_review_i == self.review_size else True

        # create input feed
        input_feed = {}
        input_feed[self.model.learning_rate.name] = learning_rate
        input_feed[self.model.user_idxs.name] = user_idxs
        input_feed[self.model.product_idxs.name] = product_idxs
        input_feed[self.model.query_word_idxs.name] = query_word_idxs
        input_feed[self.model.review_idxs.name] = review_idxs
        input_feed[self.model.word_idxs.name] = word_idxs
        if batch_context_word_idxs != None:
            for i in xrange(2 * self.model.window_size):
                input_feed[self.model.context_word_idxs[i].name] = batch_context_word_idxs[i]
        return input_feed, has_next

    def prepare_test_epoch(self, debug=False):
        self.test_user_query_set = set()
        self.test_seq = []
        for review_idx in xrange(len(self.data_set.review_info)):
            user_idx = self.data_set.review_info[review_idx][0]
            product_idx = self.data_set.review_info[review_idx][1]
            for query_idx in self.data_set.product_query_idx[product_idx]:
                if (user_idx, query_idx) not in self.test_user_query_set:
                    self.test_user_query_set.add((user_idx, query_idx))
                    self.test_seq.append((user_idx, product_idx, query_idx, review_idx))
        self.cur_uqr_i = 0

    def get_test_batch(self,debug=False):
        user_idxs, product_idxs, review_idxs, word_idxs, context_word_idxs = [],[],[],[],[]
        query_word_idxs = []
        learning_rate = self.model.init_learning_rate * max(0.0001,
                                    1.0 - self.finished_word_num / self.words_to_train)
        start_i = self.cur_uqr_i
        user_idx, product_idx, query_idx, review_idx = self.test_seq[self.cur_uqr_i]

        while len(user_idxs) < self.batch_size:
            text_list = self.data_set.review_text[review_idx]
            user_idxs.append(user_idx)
            product_idxs.append(product_idx)
            query_word_idxs.append(self.data_set.query_words[query_idx])
            review_idxs.append(review_idx)
            word_idxs.append(text_list[0])
            if self.model.need_context:
                i = 0
                start_index = i - self.model.window_size if i - self.model.window_size > 0 else 0
                context_word_list = text_list[start_index:i] + text_list[i+1:i+self.model.window_size+1]
                while len(context_word_list) < 2 * self.model.window_size:
                    context_word_list += text_list[start_index:start_index+2*self.model.window_size-len(context_word_list)]
                context_word_idxs.append(context_word_list)

            #move to the next review
            self.cur_uqr_i += 1
            if self.cur_uqr_i == len(self.test_seq):
                break
            user_idx, product_idx, query_idx, review_idx = self.test_seq[self.cur_uqr_i]

        batch_context_word_idxs = None
        length = len(word_idxs)
        if self.model.need_context:
            batch_context_word_idxs = []
            for length_idx in xrange(2 * self.model.window_size):
                batch_context_word_idxs.append(np.array([context_word_idxs[batch_idx][length_idx]
                        for batch_idx in xrange(length)], dtype=np.int64))

        has_next = False if self.cur_uqr_i == len(self.test_seq) else True
        # create input feed
        input_feed = {}
        input_feed[self.model.learning_rate.name] = learning_rate
        input_feed[self.model.user_idxs.name] = user_idxs
        input_feed[self.model.product_idxs.name] = product_idxs
        input_feed[self.model.query_word_idxs.name] = query_word_idxs
        input_feed[self.model.review_idxs.name] = review_idxs
        input_feed[self.model.word_idxs.name] = word_idxs
        if batch_context_word_idxs != None:
            for i in xrange(2 * self.model.window_size):
                input_feed[self.model.context_word_idxs[i].name] = batch_context_word_idxs[i]

        return input_feed, has_next, self.test_seq[start_i:self.cur_uqr_i]
