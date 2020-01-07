import numpy as np
import json
import random
import gzip
import math
from six.moves import range
import os

class AEMDataset:
    def __init__(self, data_path, input_train_dir, set_name):
        #get product/user/vocabulary information
        self.product_ids = []
        with gzip.open(data_path + 'product.txt.gz', 'rt') as fin:
            for line in fin:
                self.product_ids.append(line.strip())
        self.product_size = len(self.product_ids)
        self.user_ids = []
        with gzip.open(data_path + 'users.txt.gz', 'rt') as fin:
            for line in fin:
                self.user_ids.append(line.strip())
        self.user_size = len(self.user_ids)
        self.words = []
        with gzip.open(data_path + 'vocab.txt.gz', 'rt') as fin:
            for line in fin:
                self.words.append(line.strip())
        self.vocab_size = len(self.words)
        self.query_words = []
        self.query_max_length = 0
        with gzip.open(input_train_dir + 'query.txt.gz', 'rt') as fin:
            for line in fin:
                words = [int(i) for i in line.strip().split(' ')]
                if len(words) > self.query_max_length:
                    self.query_max_length = len(words)
                self.query_words.append(words)
        #pad
        for i in range(len(self.query_words)):
            self.query_words[i] = [self.vocab_size for j in range(self.query_max_length-len(self.query_words[i]))] + self.query_words[i]


        #get review sets
        self.word_count = 0
        self.vocab_distribute = np.zeros(self.vocab_size)
        self.review_info = []
        self.review_text = []
        with gzip.open(input_train_dir + set_name + '.txt.gz', 'rt') as fin:
            for line in fin:
                arr = line.strip().split('\t')
                self.review_info.append((int(arr[0]), int(arr[1]))) # (user_idx, product_idx)
                self.review_text.append([int(i) for i in arr[2].split(' ')])
                for idx in self.review_text[-1]:
                    self.vocab_distribute[idx] += 1
                self.word_count += len(self.review_text[-1])
        self.review_size = len(self.review_info)
        self.vocab_distribute = self.vocab_distribute.tolist()
        self.sub_sampling_rate = None
        self.review_distribute = np.ones(self.review_size).tolist()
        self.product_distribute = np.ones(self.product_size).tolist()
        for i, rf in enumerate(self.review_info):
            if i > 10:
                break
            print(rf)
        #get product query sets
        self.product_query_idx = []
        with gzip.open(input_train_dir + set_name + '_query_idx.txt.gz', 'rt') as fin:
            for line in fin:
                arr = line.strip().split(' ')
                query_idx = []
                for idx in arr:
                    if len(idx) < 1:
                        continue
                    query_idx.append(int(idx))
                self.product_query_idx.append(query_idx)

        # get the user review and user product set
        self.user_review_idxs = []
        self.user_product_idxs = []
        with gzip.open(os.path.join(data_path, "u_r_seq.txt.gz"), 'rt') as fin:
            for line in fin:
                reviews = [int(_) for _ in line.rstrip().split(" ")]
                self.user_review_idxs.append(reviews)
                self.user_product_idxs.append([-1 for _ in range(len( self.user_review_idxs[-1]))])

        # get user product idx data
        with gzip.open(os.path.join(data_path, "review_u_p.txt.gz"), 'rt') as fin:
            review_idx = 0
            for line in fin:
                user_idx, product_idx = int(line.rstrip().split(" ")[0]), int(line.rstrip().split(" ")[1])
                if review_idx in self.user_review_idxs[user_idx]:
                    pos = self.user_review_idxs[user_idx].index(review_idx)
                    self.user_product_idxs[user_idx][pos] = product_idx
                else:
                    print("Error: We cannot find the corresponding products")
                    pass
                review_idx += 1

        self.max_history_length = 0
        for product_idxs in self.user_product_idxs:
            if self.max_history_length < len(product_idxs):
                self.max_history_length = len(product_idxs)

        print("Data statistic: vocab %d, review %d, user %d, product %d, max history length%d\n" % (self.vocab_size,
                    self.review_size, self.user_size, self.product_size, self.max_history_length))

    def get_history_products(self, user_idx, product_idx, max_length = None):
        pos = self.user_product_idxs[user_idx].index(product_idx)
        product_history_idxs = self.user_product_idxs[user_idx][max(0, pos-max_length):pos]
        history_length = len(product_history_idxs)
        if max_length != None:
            product_history_idxs += [self.product_size for _ in range(max_length - len(product_history_idxs))]
        return product_history_idxs, history_length

    def sub_sampling(self, subsample_threshold):
        if subsample_threshold == 0.0:
            return
        self.sub_sampling_rate = [1.0 for _ in range(self.vocab_size)]
        threshold = sum(self.vocab_distribute) * subsample_threshold
        count_sub_sample = 0
        for i in range(self.vocab_size):
            #vocab_distribute[i] could be zero if the word does not appear in the training set
            self.sub_sampling_rate[i] = min((np.sqrt(float(self.vocab_distribute[i]) / threshold) + 1) * threshold / float(self.vocab_distribute[i]),
                                            1.0)
            count_sub_sample += 1

    def read_train_product_ids(self, data_path):
        self.user_train_product_set_list = [set() for i in range(self.user_size)]
        self.train_review_size = 0
        with gzip.open(data_path + 'train.txt.gz', 'rt') as fin:
            for line in fin:
                self.train_review_size += 1
                arr = line.strip().split('\t')
                self.user_train_product_set_list[int(arr[0])].add(int(arr[1]))


    def compute_test_product_ranklist(self, u_idx, original_scores, sorted_product_idxs, rank_cutoff):
        product_rank_list = []
        product_rank_scores = []
        rank = 0
        for product_idx in sorted_product_idxs:
            if product_idx in self.user_train_product_set_list[u_idx] or math.isnan(original_scores[product_idx]) or product_idx == len(self.product_ids):
                continue
            product_rank_list.append(product_idx)
            product_rank_scores.append(original_scores[product_idx])
            rank += 1
            if rank == rank_cutoff:
                break
        return product_rank_list, product_rank_scores

    def output_ranklist(self, user_ranklist_map, user_ranklist_score_map, output_path, similarity_func, debug=False):
        if debug:
            print("ranklikst length: ", len(user_ranklist_map))

        with open(output_path + 'test.'+similarity_func+'.ranklist', 'w') as rank_fout:
            for uq_pair in user_ranklist_map:
                user_id = self.user_ids[uq_pair[0]]
                for i in range(len(user_ranklist_map[uq_pair])):
                    #print(i, len(user_ranklist_map[uq_pair]))
                    #print("product_id: ", user_ranklist_map[uq_pair][i], len(self.product_ids))
                    product_id = self.product_ids[user_ranklist_map[uq_pair][i]]
                    rank_fout.write(user_id+'_'+str(uq_pair[1]) + ' Q0 ' + product_id + ' ' + str(i+1)
                            + ' ' + str(user_ranklist_score_map[uq_pair][i]) + ' ProductSearchEmbedding\n')



    def output_embedding(self, embeddings, output_file_name):
        with open(output_file_name,'w') as emb_fout:
            try:
                length = len(embeddings)
                if length < 1:
                    return
                dimensions = len(embeddings[0])
                emb_fout.write(str(length) + '\n')
                emb_fout.write(str(dimensions) + '\n')
                for i in range(length):
                    for j in range(dimensions):
                        emb_fout.write(str(embeddings[i][j]) + ' ')
                    emb_fout.write('\n')
            except:
                emb_fout.write(str(embeddings) + ' ')
