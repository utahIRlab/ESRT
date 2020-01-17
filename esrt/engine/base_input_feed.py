import abc
import typing

import tensorflow.compat.v1 as tf

class BaseInputFeed(abc.ABC):
    """

    """
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    @abc.abstractmethod
    def get_train_batch(self):
        """Defien a batch feed dictionary the model needs for training, each sub class should
           implement this method."""

    @abc.abstractmethod
    def get_test_batch(self):
        """Defien a batch feed dictionary the model needs for testing, each sub class should 
           implement this method."""
