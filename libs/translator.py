import tensorflow as tf

MAX_TOKENS = 30
class Translator(tf.Module):
    def __init__(self, tokenizer_from, tokenizer_to, transformer):
        self.tokenizer_from = tokenizer_from
        self.tokenizer_to = tokenizer_to
        self.transformer = transformer

    def __call__(self, sentence, max_length=MAX_TOKENS):
        ja_token_ids = self.tokenizer_from.encode(sentence, add_special_tokens=True)

        ja_tensor = tf.constant([ja_token_ids], dtype=tf.int64)
        encoder_input = ja_tensor
        ukrainian_start_token_id = 32000
        ukrainian_end_token_id = 32001
        start = [ukrainian_start_token_id]
        end = [ukrainian_end_token_id]

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)
            predictions = predictions[:, -1:, :]
            predicted_id = tf.argmax(predictions, axis=-1)

            predicted_id = tf.cast(predicted_id, dtype=tf.int64)

            output_array = output_array.write(i+1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())

        prep = (output)[0][1:]
        print(prep)
        prep = prep[:len(prep) - 1]
        print(prep)
        token_ids = prep.numpy().tolist()
        print(token_ids)
        text = self.tokenizer_to.decode_ids(token_ids)
        self.transformer([encoder_input, output[:,:-1]], training=False)
        return text