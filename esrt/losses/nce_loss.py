import tensorflow.compat.v1 as tf

def nce_loss(model, true_logits, sampled_logits):
    """
    Compute the nce loss with logits

    Args:
        model: esrt.engine.BaseModel.
        true_logits: tf.Tensor with shape of [batch_size, 1] with type of float32.
        sampled_logits: tf.Tensor with shape of [batch_size, num_sampled] with type of float 32.

    Returns:
        nce_loss_tensor: Tensor with shape of [batch_size, 1] with type of float 32.
    """

    # cross-entropy(logits, labels)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
    logits=true_logits, labels=tf.ones_like(true_logits))
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
    logits=sampled_logits, labels=tf.zeros_like(sampled_logits))

    # NCE-loss is the sum of the true and noise (sampled words)
    # contributions, averaged over the batch.
    nce_loss_tensor = true_xent + tf.reduce_sum(sampled_xent, 1)
    return nce_loss_tensor
