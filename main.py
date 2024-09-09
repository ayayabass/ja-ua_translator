from loader import Loader

translator = Loader().load()

sentence = 'ありがとう'

translation = translator(sentence)
print(translation)