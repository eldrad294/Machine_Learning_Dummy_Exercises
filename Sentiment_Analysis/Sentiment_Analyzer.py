import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
#
positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)', 'underpriced' ]
negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':(', 'shit', 'strange', 'threw up', 'vomit', 'overpriced' ]
neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not' ]
#
def word_feats(words):
    return dict([(word, True) for word in words])
#
positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]
#
train_set = negative_features + positive_features + neutral_features
#
classifier = NaiveBayesClassifier.train(train_set)
# Predict
neg = 0
pos = 0
sentence = "Very small but cozy restaurant. I liked cocktail menu made with a real book as a part of cover.  Cocktail was good (I asked something like mojito which wasn't on the menu) but not out of the world... I thought pasta was overpriced.... same for mashed potatoes.. I guess the flavor of food was pretty well-balanced but not very memorable..especially for the price. Complimentary bread was warm and OK."
sentence = sentence.lower()
words = sentence.split(' ')
for word in words:
    classResult = classifier.classify(word_feats(word))
    if classResult == 'neg':
        neg = neg + 1
    if classResult == 'pos':
        pos = pos + 1
#
print('Positive: ' + str(float(pos) / len(words)))
print('Negative: ' + str(float(neg) / len(words)))