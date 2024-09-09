import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

class DatasetPreparator:
    def pad_data(self, ja_tokens, ua_tokens):
        ua_inputs = [seq[:-1] for seq in ua_tokens]
        ua_labels = [seq[1:] for seq in ua_tokens]

        max_length_ja = max(len(seq) for seq in ja_tokens)
        max_length_ua_inputs = max(len(seq) for seq in ua_inputs)
        max_length_ua_labels = max(len(seq) for seq in ua_labels)

        ja_padded = tf.keras.preprocessing.sequence.pad_sequences(ja_tokens, maxlen=max_length_ja, padding='post')
        ua_inputs_padded = tf.keras.preprocessing.sequence.pad_sequences(ua_inputs, maxlen=max_length_ua_inputs, padding='post')
        ua_labels_padded = tf.keras.preprocessing.sequence.pad_sequences(ua_labels, maxlen=max_length_ua_labels, padding='post')

        ja_np = np.array(ja_padded, dtype=np.int64)
        ua_inputs_np = np.array(ua_inputs_padded, dtype=np.int64)
        ua_labels_np = np.array(ua_labels_padded, dtype=np.int64)
        return (ja_np, ua_inputs_np, ua_labels_np)
    
    def split_dataset(self, ja_np, ua_inputs_np, ua_labels_np):
        ja_train, ja_val, ua_inputs_train, ua_inputs_val, ua_labels_train, ua_labels_val = train_test_split(
            ja_np, ua_inputs_np, ua_labels_np, test_size=0.2, random_state=501
        )
        return (ja_train, ja_val, ua_inputs_train, ua_inputs_val, ua_labels_train, ua_labels_val)
    
    def preprocess_dataset(self, ja, ua_in, ua_out):
        return (tf.cast(ja, tf.int64), tf.cast(ua_in, tf.int64)), tf.cast(ua_out, tf.int64)
    
    def prefetch_dataset(self, ja_train, ja_val, ua_inputs_train, ua_inputs_val, ua_labels_train, ua_labels_val):
        train_dataset = tf.data.Dataset.from_tensor_slices(((ja_train, ua_inputs_train), ua_labels_train))
        val_dataset = tf.data.Dataset.from_tensor_slices(((ja_val, ua_inputs_val), ua_labels_val))
        train_dataset = train_dataset.map(lambda x, y: self.preprocess_dataset(x[0], x[1], y))
        val_dataset = val_dataset.map(lambda x, y: self.preprocess_dataset(x[0], x[1], y))

        batch_size = 4
        train_dataset_batches = train_dataset.batch(batch_size)
        val_dataset_batches = val_dataset.batch(batch_size)

        train_dataset_batches = train_dataset_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        val_dataset_batches = val_dataset_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        print('Train dataset batches type: ', train_dataset_batches.element_spec)
        print('Validation dataset batches type: ', val_dataset_batches.element_spec)
        return (train_dataset_batches, val_dataset_batches)