from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABC, abstractmethod


class BasicInputFeed(ABC):
    """
    This class implements a input layer for product search embedding expriments.
    """
    @abstractmethod
    def __init__(self, model, dataset, exp_settings, session):
        """
        Initialize a input feed
        Args:
            model: (BasicModel) the model we are about to training.
            dataset: (AmazonDataset) the dataset contains all necessary input information.
            exp_settings: (dict) contains some necessary (key, value) pairs for the class
            session: the current tensorflow session.
        """
        pass

    @abstractmethod
    def get_train_batch(self, batch_size):
        """
        Get a batch of training data, and it will prepare for model.step(...).

        Since the step(...) needs a list of batch-major vectors, while the dataset
        here contains many single length-major cases. The main logic here is to
        re-index the cases to be in a proper format for feeding.
        Args:
            batch_size: (int) the number of examples for each batch.
        Returns:
            input_feed: a feed dictionary for model.step(...) method.
            info_map: a dictionary contains some basic information about the batch(for debugging)
        """
        pass
