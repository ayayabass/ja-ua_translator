import tensorflow as tf
from libs.attention_mechanism.gsa import GlobalSelfAttention
from libs.attention_mechanism.feed_forward import FeedForward
from libs.embedder import PositionalEmbedding

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.self_attention.key_dim,
            'num_heads': self.self_attention.num_heads,
            'dff': self.ffn.dff,
            'dropout_rate': self.self_attention.dropout.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
  
class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
      super().__init__()
      self.num_layers = num_layers
      self.d_model = d_model

      self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
      self.enc_layers = [EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
                           for _ in range(num_layers)]
      self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
      x = self.pos_embedding(x)
      x = self.dropout(x)
      for i in range(self.num_layers):
          x = self.enc_layers[i](x)
      return x

  def get_config(self):
      config = super().get_config()
      config.update({
          'num_layers': self.num_layers,
          'd_model': self.d_model,
          'num_heads': self.enc_layers[0].self_attention.mha.get_config()['num_heads'],  # Access from config
          'dff': self.enc_layers[0].ffn.seq.layers[0].units,  # Access the units of Dense layer
          'vocab_size': self.pos_embedding.embedding.input_dim,
          'dropout_rate': self.dropout.rate,
      })
      return config

  @classmethod
  def from_config(cls, config):
      return cls(**config)
  
  def test_encoder(train_dataset_copy, input_vocab_size):
    sample_encoder = Encoder(num_layers=4,
                            d_model=512,
                            num_heads=8,
                            dff=2048,
                            vocab_size=input_vocab_size)
    for (x, y) in train_dataset_copy[0].take(1):
        print(x[0])
        sample_encoder_output = sample_encoder(x[0], training=False)

    print(sample_encoder_output.shape)