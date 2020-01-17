import tensorflow.compat.v1 as tf

from .pair_search_loss import pair_search_loss

def relation_nce_loss(model, add_weight, example_idxs, head_entity_name, relation_name, tail_entity_name):
    """
    Args:
        model: esrt.models.DREM
        add_weight: float32
        example_idxs: Tensor wish shape of [batch_size] with type of int64
        head_entity_name: str
        relation_name: str
        tail_entity_name: str

    Return:
        loss_tensor: Tensor with shape of [batch_size, 1] with type of foat32.
    """
    relation_vec = model.relation_dict[relation_name]['embedding']
    example_emb = model.entity_dict[head_entity_name]['embedding']
    label_idxs = model.relation_dict[relation_name]['idxs']
    label_emb = model.entity_dict[tail_entity_name]['embedding']
    label_bias = model.relation_dict[relation_name]['bias']
    label_size = model.entity_dict[tail_entity_name]['size']
    label_distribution = model.relation_dict[relation_name]['distribute']
    loss_tensor, embs = pair_search_loss(model, add_weight, relation_vec, example_idxs, example_emb, label_idxs, label_emb, label_bias, label_size, label_distribution)


    return model.relation_dict[relation_name]['weight'] * loss_tensor, embs
