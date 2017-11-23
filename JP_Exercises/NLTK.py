# import nltk
# nltk.download()
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, state_union
from nltk.stem import PorterStemmer
#
example_text = "Hello, how are you? My name is John Doe and I am a doctor. What is your profession? I am going to swim " \
               "if you want to come. Jenna is also coming."
#
stop_words = stopwords.words('english')
words = word_tokenize(example_text)
#
filtered_sentence = [w for w in words if not w in stop_words]
print(filtered_sentence)
#
ps = PorterStemmer()
print([ps.stem(w) for w in filtered_sentence])
#
speech_tags = []
for word in words:
    print(word)
speech_tags.append(nltk.pos_tag(words))
print(speech_tags)