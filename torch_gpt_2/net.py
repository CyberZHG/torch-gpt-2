import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_position_embedding import PositionEmbedding
from torch_layer_normalization import LayerNormalization
from torch_multi_head_attention import MultiHeadAttention
from torch_transformer import FeedForward
from torch_embed_sim import EmbeddingSim


__all__ = ['GPT_2']


def gelu(x):
    """An approximation of gelu.

    See: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1. + torch.tanh(math.sqrt(2. / math.pi) * (x + 0.044715 * torch.pow(x, 3.))))


class BlockWrapper(nn.Module):

    def __init__(self, in_features, layer):
        """Wrap layer with add and normalization.

        :param in_features: Length of input features.
        :param layer: The layer to be wrapped.
        """
        super(BlockWrapper, self).__init__()
        self.in_features = in_features
        self.layer = layer
        self.normal = LayerNormalization(normal_shape=in_features, epsilon=1e-16)

    def forward(self, x):
        return x + self.layer(self.normal(x))


class AttentionWrapper(BlockWrapper):

    def forward(self, x, mask=None):
        normal = self.normal(x)
        return x + self.layer(normal, normal, normal, mask=mask)


class EncoderComponent(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 head_num,
                 attention_activation=None,
                 feed_forward_activation=F.relu):
        """Encoder component.

        :param in_features: Length of the input features.
        :param hidden_features: Number of features inside feed-forward layer.
        :param head_num: Number of heads.
        :param attention_activation: Activation for attention layer.
        :param feed_forward_activation: Activation for feed-forward layer.
        """
        super(EncoderComponent, self).__init__()
        self.attention = AttentionWrapper(
            in_features,
            layer=MultiHeadAttention(
                in_features=in_features,
                head_num=head_num,
                activation=attention_activation,
            ),
        )
        self.feed_forward = BlockWrapper(
            in_features,
            layer=FeedForward(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=in_features,
                activation=feed_forward_activation,
            ),
        )

    def forward(self, x, mask=None):
        return self.feed_forward(self.attention(x, mask=mask))


class Encoder(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 encoder_num,
                 head_num,
                 attention_activation=None,
                 feed_forward_activation=F.relu):
        """Encoder.

        :param in_features: Length of the input features.
        :param hidden_features: Number of features inside feed-forward layer.
        :param encoder_num: Number of encoder components.
        :param head_num: Number of heads.
        :param attention_activation: Activation for attention layer.
        :param feed_forward_activation: Activation for feed-forward layer.
        """
        super(Encoder, self).__init__()
        self.components = []
        for i in range(encoder_num):
            component = EncoderComponent(
                in_features=in_features,
                hidden_features=hidden_features,
                head_num=head_num,
                attention_activation=attention_activation,
                feed_forward_activation=feed_forward_activation,
            )
            self.add_module('encoder_%d' % i, component)
            self.components.append(component)

    def forward(self, x):
        mask = MultiHeadAttention.gen_history_mask(x)
        for component in self.components:
            x = component(x, mask=mask)
        return x


class GPT_2(nn.Module):

    def __init__(self,
                 n_vocab,
                 n_ctx=1024,
                 n_embd=768,
                 n_head=12,
                 n_layer=12):
        """Get basic GPT-2 model.

        :param n_vocab: Number of vocabulary tokens.
        :param n_ctx: The length of each input.
        :param n_embd: The dimension of embeddings.
        :param n_head: Number of heads in transformer.
        :param n_layer: Number of transformer blocks.
        :return: The model.
        """
        super(GPT_2, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=n_embd)
        self.position_embedding = PositionEmbedding(num_embeddings=n_ctx, embedding_dim=n_embd)
        self.encoder = Encoder(
            in_features=n_embd,
            hidden_features=n_embd * 4,
            encoder_num=n_layer,
            head_num=n_head,
            attention_activation=None,
            feed_forward_activation=gelu,
        )
        self.layer_norm = LayerNormalization(normal_shape=n_embd)
        self.embedding_sim = EmbeddingSim(num_embeddings=n_vocab, bias=False)

    def forward(self, x):
        embed = self.position_embedding(self.embedding(x))
        encoded = self.layer_norm(self.encoder(embed))
        return self.embedding_sim(encoded, self.embedding.weight)
