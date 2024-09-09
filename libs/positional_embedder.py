import tensorflow as tf
import numpy as np

def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)

  return tf.cast(pos_encoding, dtype=tf.float32)

def apply_positional_embedding(ja_batch, ua_batch, ua_label, ja_pos_embedding, ua_pos_embedding):
            ja_pos_embedded = ja_pos_embedding(ja_batch)
            ua_pos_embedded = ua_pos_embedding(ua_batch)
            return (ja_pos_embedded, ua_pos_embedded), ua_label

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, train_dataset, val_dataset, batch_size, d_model):
    super().__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.d_model, mask_zero=True)
    

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)
  
  def fetch_datasets(train_dataset, val_dataset, batch_size):
        train_dataset_batches = train_dataset.batch(batch_size)
        val_dataset_batches = val_dataset.batch(batch_size)

        # Prefetch datasets
        train_dataset_batches = train_dataset_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        val_dataset_batches = val_dataset_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # Print dataset structure to verify
        print(train_dataset_batches)
        print(train_dataset_batches.element_spec)
        print(val_dataset_batches.element_spec)
        return [train_dataset_batches, val_dataset_batches]

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    pos_encoding = positional_encoding(length=2048, depth=self.d_model)
    x = x + pos_encoding[tf.newaxis, :length, :]
    return x
  
  