# 3rd party import
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
# stdlib import

# module import
import model_utils

def UQP_nce_loss(model, user_idxs, query_word_idxs, product_idxs, word_idxs):
	"""
    Args:
        model: (BasicModel)
        user_idxs: (tf.int32 with Shape: [batch_size, 1])
        query_word_idxs: (tf.int32 with Shape: [batch_size, max_query_length])
        product_idxs: (tf.int32 with Shape: [batch_size, 1])
        word_idxs: (tf,int32 with Shape: [batch_size, 1])
    Return:
        UQP loss: (tf.float) See paper: Qingyao Ai, Yongfeng Zhang, Keping Bi, Xu Chen, W. Bruce Croft. 2017.
                                        Learning a Hierarchical Embedding Model for Personalized ProductSearch. In Proceedings of SIGIR ’17
    """
	batch_size = array_ops.shape(word_idxs)[0]#get batch_size
	loss = None
	regularization_terms = []

	#word prediction loss
	uw_loss, uw_embs = single_nce_loss(model,user_idxs, model.user_emb, word_idxs, model.word_emb,
					model.word_bias, model.vocab_size, model.vocab_distribute)
	pw_loss, pw_embs = single_nce_loss(model,product_idxs, model.product_emb, word_idxs, model.word_emb,
					model.word_bias, model.vocab_size, model.vocab_distribute)
	loss = uw_loss + pw_loss
	regularization_terms += uw_embs + pw_embs

	# get query vector
	query_vec, qw_embs = model_utils.get_query_embedding(model, query_word_idxs, None)
	regularization_terms += qw_embs

	#product prediction loss
	uqr_loss, uqr_embs = pair_search_loss(model, query_vec, user_idxs, model.user_emb, product_idxs, model.product_emb,
						model.product_bias, model.product_size, model.product_distribute)
	regularization_terms += uqr_embs
	loss += uqr_loss

	# L2 regularization
	if model.hparams.L2_lambda > 0:
		l2_loss = tf.nn.l2_loss(regularization_terms[0])
		for i in range(1,len(regularization_terms)):
			l2_loss += tf.nn.l2_loss(regularization_terms[i])
		loss += model.hparams.L2_lambda * l2_loss

	return loss / math_ops.cast(batch_size, dtypes.float32)

def pair_search_loss(model, query_vec, example_idxs, example_emb, label_idxs, label_emb, label_bias, label_size, label_distribution):

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
	example_vec = tf.nn.embedding_lookup(example_emb, example_idxs) * (1-model.Wu) + query_vec * model.Wu

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

def single_nce_loss(model, example_idxs, example_emb, label_idxs, label_emb, label_bias, label_size, label_distribution):
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
	example_vec = tf.nn.embedding_lookup(example_emb, example_idxs)

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
	#return model.nce_loss(true_logits, true_logits)

def nce_loss(model, true_logits, sampled_logits):
	"Build the graph for the NCE loss."

	# cross-entropy(logits, labels)
	true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
			logits=true_logits, labels=tf.ones_like(true_logits))
	sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
			logits=sampled_logits, labels=tf.zeros_like(sampled_logits))

	# NCE-loss is the sum of the true and noise (sampled words)
	# contributions, averaged over the batch.
	nce_loss_tensor = (tf.reduce_sum(true_xent) + tf.reduce_sum(sampled_xent))
	return nce_loss_tensor
