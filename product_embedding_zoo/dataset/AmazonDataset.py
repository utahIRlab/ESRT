import numpy as np
import gzip
from six.moves import xrange

class AmazonDataset:
    def __init__(self, data_path, input_train_dir, set_name):
        """
        Construct data structures to represent amazon dataset and prep for input feed layer.
        Args:
            data_path: The data directory.
            input_train_dir: the directory contains train and test dataset.
            set_name: set "train" to get train dataset, and "test" to get test dataset.
        """
        # get basic information for user/product/words
        self.user_ids, self.product_ids, self.words = [], [], []
        self.user_size, self.product_size, self.vocab_size = 0, 0, 0
        with gzip.open(data_path + 'users.txt.gz', 'r') as fin:
            for line in fin:
                self.user_ids.append(line.decode('utf-8').strip())
        self.user_size = len(self.user_ids)
        with gzip.open(data_path + 'product.txt.gz', 'r') as fin:
            for line in fin:
                self.product_ids.append(line.decode('utf-8').strip())
        self.product_size = len(self.product_ids)
        with gzip.open(data_path + 'vocab.txt.gz', 'r') as fin:
            for line in fin:
                self.words.append(line.decode('utf-8').strip())
        self.vocab_size = len(self.words)

        # get query infomration
        self.query_words = []
        self.query_max_length = 0
        with gzip.open(input_train_dir + 'query.txt.gz', 'r') as fin:
            for line in fin:
                words = [int(i) for i in line.decode('utf-8').rstrip().split(' ')]
                if len(words) > self.query_max_length:
                    self.query_max_length = len(words)
                self.query_words.append(words)
        # pad
        for i in xrange(len(self.query_words)):
            self.query_words[i] = [self.vocab_size for j in xrange(self.query_max_length-len(self.query_words[i]))] + self.query_words[i]

        # get review set
        self.vocab_distribute = np.zeros(self.vocab_size)
        self.word_count, self.review_size = 0, 0
        self.review_info = []
        self.review_text = []
        with gzip.open(input_train_dir + set_name + '.txt.gz', 'r') as fin:
            for line in fin:
                arr = line.decode('utf-8').rstrip().split('\t')
                self.review_info.append((int(arr[0]), int(arr[1]))) #(user_idx, product_idx)
                self.review_text.append([int(i) for i in arr[2].split(' ')])
                # get review information
                for i in self.review_text[-1]:
                    self.vocab_distribute[i] += 1
                self.word_count += len(self.review_text[-1])
            self.review_size = len(self.review_text)
            self.vocab_distribute = self.vocab_distribute.tolist()
            self.review_distribute = np.ones(self.review_size).tolist()
            self.product_distribute = np.ones(self.product_size).tolist()
            self.sub_sampling_rate = None

        #get product query sets
        self.product_query_idx = []
        with gzip.open(input_train_dir + set_name + '_query_idx.txt.gz', 'r') as fin:
            for line in fin:
                arr = line.decode('utf-8').strip().split(' ')
                query_idx = []
                for idx in arr:
                    if len(idx) < 1:
                        continue
                    query_idx.append(int(idx))
                self.product_query_idx.append(query_idx)

        print("Data statistic: vocab %d, review %d, user %d, product %d\n" % (self.vocab_size,
                    self.review_size, self.user_size, self.product_size))
    def sub_sampling(self, subsample_threshold):
        """Intuitively the word with extremely large frequency tends to have low
        subsampling rate.
        """
        if subsample_threshold == 0.0:
            return

        self.sub_sampling_rate = [1.0 for _ in xrange(self.vocab_size)]
        threshold = sum(self.vocab_distribute) * subsample_threshold
        count_sub_sample = 0
        for i in xrange(self.vocab_size):
            #vocab_distribute[i] could be zero if the word does not appear in the training set
            self.sub_sampling_rate[i] = min((np.sqrt(float(self.vocab_distribute[i]) / threshold) + 1) * threshold / float(self.vocab_distribute[i]),
                                            1.0)
            count_sub_sample += 1
