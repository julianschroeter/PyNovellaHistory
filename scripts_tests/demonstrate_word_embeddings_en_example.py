# Imports
from preprocessing.presetting import language_model_path, vocab_lists_dicts_directory, save_stoplist, keywords_to_semantic_fields, load_stoplist
import os
import spacy
import numpy as np

system = "my_xps" # "wcph113"

my_language_model_en = 'en_core_web_lg' #"de_dep_news_trf"

#language_model_path(system)

print(my_language_model_en)

vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" )

lists_path = vocab_lists_dicts_directory(system)
list_of_keywords=["honor", "guilt", "shame", "crime", "justice", "court"]

nlp = spacy.load(my_language_model_en)
all_output_words, all_vectors = [], []


# for word in all_output_words:
for word in list_of_keywords:
    print(nlp.vocab.has_vector(word))
    new_vector = nlp.vocab.get_vector(word)
    all_vectors.append(new_vector)
    print(new_vector)

print(all_output_words)
print(len(all_output_words))
print(len(all_vectors))

vectors = np.stack(all_vectors, axis=0)
v = vectors
new_vec = v[0] - v[1] + v[2]

ms = nlp.vocab.vectors.most_similar(np.asarray([new_vec]), n=3)
new_words = [nlp.vocab.strings[w] for w in ms[0][0]]
print(new_words)

all_output_words = list_of_keywords.copy()
# all_output_words.extend(new_words)

all_vectors = []
for word in all_output_words:
    new_vector = nlp.vocab.get_vector(word)
    all_vectors.append(new_vector)
vectors = np.stack(all_vectors, axis=0)

all_output_words = list(set(all_output_words))


#all_output_words.extend(["sum_vector"])
print(all_output_words)


#colors_list = ["red" if word in compl_words else "blue" if word in list_of_keywords else "green" for word in all_output_words]
colors_list = ["blue" if word in list_of_keywords else ("red" if word in list_of_keywords else "green") for word in all_output_words]

print(colors_list)

print(vectors.shape)
#new_vec = new_vec.T
print(np.asarray(new_vec).shape)

#vectors = np.append(vectors, np.asarray([new_vec]), axis=0)
print(vectors.shape)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
vecs = pca.fit_transform(vectors)

plt.scatter(vecs[:,0], vecs[:,1], c=colors_list)
#for i, name in enumerate(all_output_words):
for i, name in enumerate(all_output_words):
    plt.annotate(name, vecs[i])
plt.show()


fig = plt.figure(figsize=[5,5])
ax = plt.axes(projection="3d")

pca = PCA(n_components=3)
vecs = pca.fit_transform(vectors)
print(vecs)


for i, name in enumerate(all_output_words):

    ax.scatter3D(vecs[i, 0], vecs[i, 1], vecs[i, 2], c=colors_list[i])
    ax.text(vecs[i, 0], vecs[i, 1], vecs[i, 2], '%s' % (str(name)), size=10, zorder=1,
            color=colors_list[i])

plt.title("Word embedding")
plt.show()

#semantic_words_list = keywords_to_semantic_fields(list_of_keywords=["erschrecken"], n_most_relevant_words=50,
 #                                         spacy_model=my_language_model_de, vocabulary_path=os.path.join(vocab_lists_dicts_directory(system), "wordlist_german.txt" ))


#print(semantic_words_list)