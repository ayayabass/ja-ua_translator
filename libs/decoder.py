import tensorflow as tf
from libs.attention_mechanism.csa import CausalSelfAttention
from libs.attention_mechanism.ca import CrossAttention
from libs.attention_mechanism.feed_forward import FeedForward
from libs.embedder import PositionalEmbedding

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.causal_self_attention = CausalSelfAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
        self.cross_attention = CrossAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.ffn(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.causal_self_attention.key_dim,
            'num_heads': self.causal_self_attention.num_heads,
            'dff': self.ffn.dff,
            'dropout_rate': self.causal_self_attention.dropout.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
  
class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
                           for _ in range(num_layers)]
        self.last_attn_scores = None

    def call(self, x, context):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.dec_layers[0].causal_self_attention.mha.get_config()['num_heads'],  # Access from config
            'dff': self.dec_layers[0].ffn.seq.layers[0].units,  # Access the units of Dense layer
            'vocab_size': self.pos_embedding.embedding.input_dim,
            'dropout_rate': self.dropout.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
  
    def test_decoder(train_dataset, train_dataset_copy, output_vocab_size):
      sample_decoder = Decoder(num_layers=4,
                          d_model=512,
                          num_heads=8,
                          dff=2048,
                          vocab_size=output_vocab_size)
      for (p, y) in train_dataset.take(1):
          for (k, n) in train_dataset_copy[0].take(1):
              output = sample_decoder(
                  x=n,
                  context=p[0])

      print(output.shape)
      sample_decoder.last_attn_scores.shape