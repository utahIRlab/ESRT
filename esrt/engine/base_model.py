import abc
import typing

import tensorflow.compat.v1 as tf

from esrt.engine.param_table import ParamTable

class BaseModel(abc.ABC):
    def __init__(self, dataset, params: ParamTable, forward_only: bool = True):
        """
        Args:

        Usage:

        """
        self._dataset = dataset
        self._params = params
        self._forward_only = forward_only

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params: ParamTable):
        self._params = new_params

    @abc.abstractmethod
    def build(self):
        """Build the model, each sub class need to implement this method."""

    @abc.abstractmethod
    def _build_placeholder(self):
        """Define all placeholders the model needs, each sub class need to
            implement this private class."""
    @abc.abstractmethod
    def _build_embedding_graph_and_loss(self, scope=None):
        """Define the computation graph, and output a tf.Tensor loss,
           each sub class need to implement this private method."""

    @abc.abstractmethod
    def _build_optimizer(self):
        """Define the optimizer, and return the 'update' operation, each
            sub class need to implement this private class"""

    @abc.abstractmethod
    def step(self, session ,input_feed, forward_only=False, test_mode=None):
        """
        train or test for one batch, each sub class need to implement this method.

        Args:
            session: tf.Session instance.
            input_feed:  dictionary.
                        its key with type of string, is name of a placeholder.
                        its value with type of tf.Tensor.
            forward_only:  bool value, is False when training, is True when testing
            test_mode: string.
        """
