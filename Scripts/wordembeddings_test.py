# Imports
from scipy.spatial import distance
import spacy
from Preprocessing.Presetting import language_model_path
import numpy as np
from collections import Counter

system = "wcph113"
# Load the spacy vocabulary
my_language_model_de = language_model_path(system)
nlp = spacy.load(my_language_model_de)


# Format the input vector for use in the distance function
# In this case we will artificially create a word vector from a real word ("frog")
# but any derived word vector could be used



input_words = ["Gespenst", "Geister", "Geisterwesen", "Geisterwelt"]

all_output_words = []

for word in input_words:
    ms = nlp.vocab.vectors.most_similar(
        np.asarray([nlp.vocab.vectors[nlp.vocab.strings[word]]]), n=50)
    words = [nlp.vocab.strings[w] for w in ms[0][0]]
    print(words)
    all_output_words.extend(words)

all_words_string = " ".join(all_output_words)

doc = nlp(all_words_string)

lemma_list = [token.lemma_ for token in doc]

print("Lemma Liste :", lemma_list)
word_counter = Counter(lemma_list)

print(word_counter)

words_list = [word for word, count in word_counter.items() if count >= 2]

print(words_list)


input_words = ["Schauder", "unheimlich", "furchtbar", "zittern", "Entsetzen"]
all_output_words = []

for word in input_words:
    ms = nlp.vocab.vectors.most_similar(
        np.asarray([nlp.vocab.vectors[nlp.vocab.strings[word]]]), n=50)
    words = [nlp.vocab.strings[w] for w in ms[0][0]]
    print(words)
    all_output_words.extend(words)

all_words_string = " ".join(all_output_words)

doc = nlp(all_words_string)

lemma_list = [token.lemma_ for token in doc]

print("Lemma Liste :", lemma_list)
word_counter = Counter(lemma_list)

print(word_counter)

words_list = [word for word, count in word_counter.items() if count >= 2]

print("20 häufigste: ", word_counter.most_common(20))

print(words_list)