import tensorflow.compat.v1 as tf
from tensorflow.python.ops import array_ops

from esrt.losses import nce_loss

def pair_search_loss(model, add_weight, query_vec, example_idxs, example_emb, label_idxs, label_emb,
                    label_bias, label_size, label_distribution):
    """
    Args:
        model: esrt.engine.BaseModel.
        query_ve: Tensor with shape of [batch_size, embed_size].
        example_idxs: Tensor with shape of [batch_size] with type of int32.
        example_emb: Tensor with shape of [id_size, embed_size] with type of float32.
        label_idxs: Tensor with shape of [batch_size] with type of int32.
        label_embd: Tensor with shape of [id_size, embed_size] with type of float32.
        label_bias: Tensor with shape of [id_size] with type of float32
        label_size: int32, its value
        label_distribution: A list with type of float32

    Return:
        loss_tensor: Tensor with shape of [batch_size, 1] with type of float32.
    """
    batch_size = array_ops.shape(example_idxs)[0]#get batch_size
    # Nodes to compute the nce loss w/ candidate sampling.
    labels_matrix = tf.reshape(tf.cast(label_idxs,dtype=tf.int64),[batch_size, 1])

    # Negative sampling.
    sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=model.negative_sample,
            unique=False,
            range_max=label_size,
            distortion=0.75,
            unigrams=label_distribution))

    #get example embeddings [batch_size, embed_size]
    example_vec = tf.nn.embedding_lookup(example_emb, example_idxs) * (1-add_weight) + query_vec * add_weight

    #get label embeddings and bias [batch_size, embed_size], [batch_size, 1]
    true_w = tf.nn.embedding_lookup(label_emb, label_idxs)
    true_b = tf.nn.embedding_lookup(label_bias, label_idxs)

    #get sampled embeddings and bias [num_sampled, embed_size], [num_sampled, 1]
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
