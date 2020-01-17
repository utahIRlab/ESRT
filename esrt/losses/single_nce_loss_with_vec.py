import tensorflow.compat.v1 as tf
from tensorflow.python.ops import array_ops

from .nce_loss import nce_loss

def single_nce_loss_with_vec(model, example_vec, label_idxs, label_emb, label_bias, label_size, label_distribution):
    """Combine nce loss with negative samples.
    Args:
        model: the model that contains all the functions and parameters.
        example_vec: the example vectors.
        label_idxs: the index of the label vectors.
        label_emb: the embedding matrix of the example vectors.
        label_bias: the bias matrix of the example vectors.
        label_size: the number of all possible labels.
        label_distribution: the distribution used for sampling labels.

    Returns:
        Aggregated nce loss and the parameters for L2 regularization.

    """
    batch_size = array_ops.shape(label_idxs)[0]  # get batch_size
    # Nodes to compute the nce loss w/ candidate sampling.
    labels_matrix = tf.reshape(tf.cast(label_idxs, dtype=tf.int64), [batch_size, 1])

    # Negative sampling.
    sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels_matrix,
        num_true=1,
        num_sampled=model.negative_sample,
        unique=False,
        range_max=label_size,
        distortion=0.75,
        unigrams=label_distribution))

    # get label embeddings and bias [batch_size, embed_size], [batch_size, 1]
    true_w = tf.nn.embedding_lookup(label_emb, label_idxs)
    true_b = tf.nn.embedding_lookup(label_bias, label_idxs)

    # get sampled embeddings and bias [num_sampled, embed_size], [num_sampled, 1]
    sampled_w = tf.nn.embedding_lookup(label_emb, sampled_ids)
    sampled_b = tf.nn.embedding_lookup(label_bias, sampled_ids)

    # True logits: [batch_size, 1]
    true_logits = tf.reduce_sum(tf.multiply(example_vec, true_w), 1) + true_b

    # Sampled logits: [batch_size, num_sampled]
    # We replicate sampled noise lables for all examples in the batch
    # using the matmul.
    sampled_b_vec = tf.reshape(sampled_b, [model.negative_sample])
    sampled_logits = tf.matmul(example_vec, sampled_w, transpose_b=True) + sampled_b_vec

    return nce_loss(model, true_logits, sampled_logits), [example_vec, true_w, sampled_w]
