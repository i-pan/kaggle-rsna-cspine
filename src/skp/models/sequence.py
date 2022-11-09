import numpy as np
import torch
import torch.nn as nn

from transformers.models.distilbert.modeling_distilbert import Transformer as T
from .pooling import create_pool1d_layer


class Config:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Transformer(nn.Module):
    """
    If predict_sequence is True, then the model will predict an output
    for each element in the sequence. If False, then the model will
    predict a single output for the sequence. 
    
    e.g., classifying each image in a CT scan vs the entire CT scan
    """
    def __init__(self,
                 num_classes,
                 embedding_dim=512,
                 hidden_dim=1024,
                 n_layers=4,
                 n_heads=16,
                 dropout=0.2,
                 attention_dropout=0.1,
                 output_attentions=False,
                 activation='gelu',
                 output_hidden_states=False,
                 chunk_size_feed_forward=0,
                 predict_sequence=True,
                 pool=None
    ):
        super().__init__()
        config = Config(**{
                'dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'n_layers': n_layers,
                'n_heads': n_heads,
                'dropout': dropout,
                'attention_dropout': attention_dropout,
                'output_attentions': output_attentions,
                'activation': activation,
                'output_hidden_states': output_hidden_states,
                'chunk_size_feed_forward': chunk_size_feed_forward
            })

        self.transformer = T(config)
        self.predict_sequence = predict_sequence

        if not predict_sequence:
            if isinstance(pool, str):
                self.pool_layer = create_pool1d_layer(pool)
                if pool == "catavgmax":
                    embedding_dim *= 2
            else:
                self.pool_layer = nn.Identity()

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def extract_features(self, x):
        x, mask = x
        x = self.transformer(x, attn_mask=mask, head_mask=[None]*x.size(1))
        x = x[0]

        if not self.predict_sequence:
            if isinstance(self.pool_layer, nn.Identity):
                # Just take the last vector in the sequence
                x = x[:, 0]
            else:
                x = self.pool_layer(x.transpose(-1, -2))

        return x

    def classify(self, x):

        if not self.predict_sequence:
            if isinstance(self.pool_layer, nn.Identity):
                # Just take the last vector in the sequence
                x = x[:, 0]
            else:
                x = self.pool_layer(x.transpose(-1, -2))

        out = self.classifier(x)

        if self.classifier.out_features == 1:
            return out[..., 0]
        else:
            return out

    def forward_tr(self, x, mask):
        output = self.transformer(x, attn_mask=mask, head_mask=[None]*x.size(1))
        return self.classify(output[0])

    def forward(self, x):
        x, mask = x
        return self.forward_tr(x, mask)


class DualTransformer(nn.Module):
    """
    Essentially the same as above except predicts both sequence and study labels.
    """
    def __init__(self,
                 num_classes,
                 embedding_dim=512,
                 hidden_dim=1024,
                 n_layers=4,
                 n_heads=16,
                 dropout=0.2,
                 attention_dropout=0.1,
                 output_attentions=False,
                 activation='gelu',
                 output_hidden_states=False,
                 chunk_size_feed_forward=0,
                 pool=None
    ):
        super().__init__()
        config = Config(**{
                'dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'n_layers': n_layers,
                'n_heads': n_heads,
                'dropout': dropout,
                'attention_dropout': attention_dropout,
                'output_attentions': output_attentions,
                'activation': activation,
                'output_hidden_states': output_hidden_states,
                'chunk_size_feed_forward': chunk_size_feed_forward
            })

        self.transformer = T(config)

        self.pool_layer = nn.Identity()

        self.classifier1 = nn.Linear(embedding_dim, num_classes)
        self.classifier2 = nn.Linear(embedding_dim, num_classes)

    def classify(self, x):

        if isinstance(self.pool_layer, nn.Identity):
            # Just take the last vector in the sequence
            x_summ = x[:, 0]
        else:
            x_summ = self.pool_layer(x.transpose(-1, -2))

        # Element-wise labels
        out1 = self.classifier1(x)[:, :, 0]
        # Single label for whole sequence
        out2 = self.classifier2(x_summ)
        out = torch.cat([out1, out2], dim=1)
        return out

    def forward_tr(self, x, mask):
        output = self.transformer(x, attn_mask=mask, head_mask=[None]*x.size(1))
        return self.classify(output[0])

    def forward(self, x):
        x, mask = x
        return self.forward_tr(x, mask)


class DualTransformerV2(nn.Module):
    """
    More complicated variant of DualTransformer.
    Returns tuple of: (element-wise prediction, sequence prediction)
    """
    def __init__(self,
                 num_seq_classes,
                 num_classes,
                 embedding_dim=512,
                 hidden_dim=1024,
                 n_layers=4,
                 n_heads=16,
                 dropout=0.2,
                 attention_dropout=0.1,
                 output_attentions=False,
                 activation='gelu',
                 output_hidden_states=False,
                 chunk_size_feed_forward=0,
                 pool=None
        ):
        super().__init__()
        config = Config(**{
                'dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'n_layers': n_layers,
                'n_heads': n_heads,
                'dropout': dropout,
                'attention_dropout': attention_dropout,
                'output_attentions': output_attentions,
                'activation': activation,
                'output_hidden_states': output_hidden_states,
                'chunk_size_feed_forward': chunk_size_feed_forward
            })

        self.transformer = T(config)

        self.pool_layer = nn.Identity()

        self.classifier1 = nn.Linear(embedding_dim, num_seq_classes)
        self.classifier2 = nn.Linear(embedding_dim, num_classes)

    def classify(self, x):

        if isinstance(self.pool_layer, nn.Identity):
            # Just take the last vector in the sequence
            x_summ = x[:, 0]
        else:
            x_summ = self.pool_layer(x.transpose(-1, -2))

        # Element-wise labels
        out1 = self.classifier1(x)
        if self.classifier1.out_features == 1:
            out1 = out1[:, :, 0]
        # Single label for whole sequence
        out2 = self.classifier2(x_summ)
        return out1, out2

    def forward_tr(self, x, mask):
        output = self.transformer(x, attn_mask=mask, head_mask=[None]*x.size(1))
        return self.classify(output[0])

    def forward(self, x):
        x, mask = x
        return self.forward_tr(x, mask)

