from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from abc import ABC, abstractmethod


class BasicModel(ABC):
    """
    The basic class that contains all the API needed for the product embedding model
    """
    @abstractmethod
    def __init__(self, dataset, exp_settings, forward_only):
        """
        Create the model
        Args:
            dataset: (AmazonDataset) the dataset for building input layer.
            exp_settings: (dictionary) the dictionary contains model setting.
            foward_only: (bool) set true to conduct prediction, false to training.
        """
        pass

    @abstractmethod
    def step(self, session, input_feed, forward_only, test_mode):
        """
        Run a step of model given proper input_feed.
        Args:
            session: (tf.Session) tensorflow session to train.
            input_feed: (dictionary) the dictoinary contains all input feed data.
            foward_only: (bool) set true to conduct prediction, false to training.
            test_mode: (str) the mode for step. It could be 'product_scores', 'output_embedding', etc.
        Return:
            A triple consist of loss, outputs(None if we do backward), and a tf.summary
            containing the related information about the step.
        """
        pass
