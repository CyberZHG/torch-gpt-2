import json
import torch
import torch.nn as nn
import tensorflow as tf
from .net import GPT_2


__all__ = ['load_trained_model_from_checkpoint']


def weight_loader(checkpoint_path):
    def _loader(name, transpose=False):
        weight = torch.from_numpy(tf.train.load_variable(checkpoint_path, name)).type(torch.get_default_dtype())
        if len(weight.size()) == 3:
            _, h, w = weight.size()
            weight = weight.view(h, w)
        if transpose:
            weight = weight.transpose(1, 0)
        return weight
    return _loader


def load_trained_model_from_checkpoint(config_path,
                                       checkpoint_path,
                                       seq_len=None):
    """Load trained official model from checkpoint.

    :param config_path: The path to the JSON configuration file. (hparams.json)
    :param checkpoint_path: The path to the checkpoint files, should end with '.ckpt'.
    :param seq_len: If it is not None and it is shorter than the value in the config file, the weights in
                    position embeddings will be sliced to fit the new length.
    :return: The model.
    """
    with open(config_path, 'r') as reader:
        config = json.load(reader)
    loader = weight_loader(checkpoint_path)
    if seq_len is None:
        n_ctx = config['n_ctx']
    else:
        n_ctx = min(seq_len, config['n_ctx'])
    n_embd = config['n_embd']

    net = GPT_2(
        n_vocab=config['n_vocab'],
        n_ctx=n_ctx,
        n_embd=n_embd,
        n_head=config['n_head'],
        n_layer=config['n_layer'],
    )

    net.embedding.weight = nn.Parameter(loader('model/wte:0'))
    net.position_embedding.weight = nn.Parameter(loader('model/wpe:0')[:seq_len, :])
    for i in range(config['n_layer']):
        layer = net.encoder.components[i]

        layer.attention.normal.gamma = nn.Parameter(loader('model/h%d/ln_1/g:0' % i))
        layer.attention.normal.beta = nn.Parameter(loader('model/h%d/ln_1/b:0' % i))

        attention_weight = loader('model/h%d/attn/c_attn/w:0' % i, True)
        attention_bias = loader('model/h%d/attn/c_attn/b:0' % i)
        layer.attention.layer.linear_q.weight = nn.Parameter(attention_weight[:n_embd, :])
        layer.attention.layer.linear_q.bias = nn.Parameter(attention_bias[:n_embd])
        layer.attention.layer.linear_k.weight = nn.Parameter(attention_weight[n_embd:-n_embd, :])
        layer.attention.layer.linear_k.bias = nn.Parameter(attention_bias[n_embd:-n_embd])
        layer.attention.layer.linear_v.weight = nn.Parameter(attention_weight[-n_embd:, :])
        layer.attention.layer.linear_v.bias = nn.Parameter(attention_bias[-n_embd:])
        layer.attention.layer.linear_o.weight = nn.Parameter(loader('model/h%d/attn/c_proj/w:0' % i, True))
        layer.attention.layer.linear_o.bias = nn.Parameter(loader('model/h%d/attn/c_proj/b:0' % i))

        layer.feed_forward.normal.gamma = nn.Parameter(loader('model/h%d/ln_2/g:0' % i))
        layer.feed_forward.normal.beta = nn.Parameter(loader('model/h%d/ln_2/b:0' % i))

        layer.feed_forward.layer.linear_h.weight = nn.Parameter(loader('model/h%d/mlp/c_fc/w:0' % i, True))
        layer.feed_forward.layer.linear_h.bias = nn.Parameter(loader('model/h%d/mlp/c_fc/b:0' % i))
        layer.feed_forward.layer.linear_o.weight = nn.Parameter(loader('model/h%d/mlp/c_proj/w:0' % i, True))
        layer.feed_forward.layer.linear_o.bias = nn.Parameter(loader('model/h%d/mlp/c_proj/b:0' % i))

    net.layer_norm.gamma = nn.Parameter(loader('model/ln_f/g:0'))
    net.layer_norm.beta = nn.Parameter(loader('model/ln_f/b:0'))

    return net
