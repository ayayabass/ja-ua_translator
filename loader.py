import tensorflow as tf
import libs.tokenizer_utils as tu
import libs.plotter as plotter

from libs.dataset_preparator import DatasetPreparator
from libs.data_loader import DataLoader
from libs.transformer import Transformer
from libs.learning_scheduler import CustomSchedule
from libs.loss_acc_functions.loss import masked_loss
from libs.loss_acc_functions.acc import masked_accuracy
from libs.translator import Translator

class Loader:
    def train(self, input_file, output_file):
        ja_data, ua_data = DataLoader().load_data(input_file, output_file)
        ja_tokenizer, ua_tokenizer = tu.create_ja_tokenizer(), tu.create_ua_tokenizer()
        ja_tokens, ua_tokens = tu.tokenize(ja_tokenizer, 'ja', ja_data), tu.tokenize(ua_tokenizer, 'ua', ua_data)
        plotter.plot_data(ja_tokens, ua_tokens)
        dp = DatasetPreparator()
        ja_padded, ua_inputs_padded, ua_labels_padded = dp.pad_data(ja_tokens, ua_tokens)
        ja_train, ja_val, ua_inputs_train, ua_inputs_val, ua_labels_train, ua_labels_val = dp.split_dataset(ja_padded, ua_inputs_padded, ua_labels_padded)
        train_dataset, val_dataset = dp.prefetch_dataset(ja_train, ja_val, ua_inputs_train, ua_inputs_val, ua_labels_train, ua_labels_val)
        ja_vocab_size = ja_tokenizer.vocab_size
        ua_vocab_size = 32002
        train_dataset_copy = (train_dataset, 1)
        val_dataset_copy = (val_dataset, 1)

        #hyperparameters
        num_layers = 4
        d_model = 128
        dff = 512
        num_heads = 8
        dropout_rate = 0.1

        transformer = Transformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=ja_vocab_size,
            target_vocab_size=ua_vocab_size,
            dropout_rate=dropout_rate)
        
        learning_rate = CustomSchedule(d_model)
        learning_rate.print_learning_rate(learning_rate, 40000)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
        
        transformer.compile(
            loss=masked_loss,
            optimizer=optimizer,
            metrics=[masked_accuracy])
        
        transformer.fit(train_dataset_copy[0],
            epochs=1,
            validation_data=val_dataset_copy[0]
        )
        translator = Translator(ja_tokenizer, ua_tokenizer, transformer)

        #SAVE METHOD
        #transformer.save('/new_ja_ua_translator/data/transformer')
        transformer.save_weights('./data/transformer_weights.h5')
        return translator
    
    def load(self):
        try:
            new_transformer = Transformer(
            num_layers=4,
            d_model=128,
            num_heads=8,
            dff=512,
            input_vocab_size=32000,
            target_vocab_size=32002,
            dropout_rate=0.1)

            dummy_context = tf.random.uniform((1, 10))
            dummy_x = tf.random.uniform((1, 10))
            _ = new_transformer((dummy_context, dummy_x))
            
            new_transformer.load_weights('./data/transformer_weights.h5')
            new_translator = Translator(tu.create_ja_tokenizer(), tu.create_ua_tokenizer(), new_transformer)
            return new_translator
        except OSError:
            new_translator = self.train('./MultiCCAligned.ja.txt', './MultiCCAligned.uk.txt')
            return new_translator