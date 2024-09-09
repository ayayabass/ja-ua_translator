import sentencepiece as spm
from transformers import BertJapaneseTokenizer

def create_ua_tokenizer():
    try:
        sp = spm.SentencePieceProcessor(model_file='./ukr_model.model')
    except OSError:
        spm.SentencePieceTrainer.train(input='./MultiCCAligned.uk.txt', model_prefix='ukr_model', vocab_size=32000)
        sp = spm.SentencePieceProcessor(model_file='./ukr_model.model')
    return sp

def create_ja_tokenizer():
    return BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

def tokenize(tokenizer, type, data):
        if type == 'ja':
            words = [tokenizer.tokenize(sentence) for sentence in data]
            tokens = [tokenizer.encode(sentence)for sentence in data]
        elif type == 'ua':
            words = [tokenizer.tokenize(sentence, out_type=str) for sentence in data]
            tokens = [[30000] + tokenizer.encode(sentence) + [30001] for sentence in data] 
        print('Length of words: ', len(words))
        print('Words example: ', words[0])
        print('Tokens example: ', tokens[0])
        print('Max length of tokens: ', max(len(seq) for seq in tokens))
        
        return tokens