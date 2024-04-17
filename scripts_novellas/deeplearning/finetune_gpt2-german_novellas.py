system = "wcph113" # "my_mac"
if system == "wcph113":
    import sys
    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import os
from preprocessing.deeplearning import NovellaDataset, train, text_generation

os.environ["CUDA_VISIBLE_DEVICES"]="2"

print(torch.cuda.is_available())


system = "wcph113"

from preprocessing.presetting import local_temp_directory, global_corpus_representation_directory
import os
import pandas as pd
from preprocessing.sampling import select_from_corpus_df

# This script selects the Novellen and Erzählungen with the highest predictive probabilites on
# validation sets in bootstrapped classification tasks (based on log reg).
# it returns chunks of the typical Novellen

pred_probs_path = os.path.join(local_temp_directory(system), "av_pred_probabs_N-E_n100.csv")

pred_probs_df = pd.read_csv(pred_probs_path, index_col=0)
df = pred_probs_df
print(df)

df_novellas = df[df["Gattungslabel_ED_normalisiert"] == "N"]
print(df_novellas)
typical_novellas_ids = df_novellas[df_novellas["mean"] > 0.6].index.tolist()

print(typical_novellas_ids)

sentences_df = pd.read_csv(os.path.join(global_corpus_representation_directory(system), "novella_corpus_as_sentences_for_BERT-class.csv"), index_col=0)

print(sentences_df)

N_sent_samples = select_from_corpus_df(sentences_df,15, typical_novellas_ids, id_cat="doc_id")

print(N_sent_samples)

print(N_sent_samples.sent_string)

training_data = N_sent_samples["sent_string"]
print(training_data)

dataset = NovellaDataset(training_data, truncate=True, gpt2_type="gpt2")

print(dataset.data)


#Get the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
model = GPT2LMHeadModel.from_pretrained('dbmdz/german-gpt2')

#Accumulated batch size (since GPT2 is so big)


trained_model = train(dataset, model, tokenizer)





# Run the functions to generate the lyrics

generated_novellas = text_generation(["Elende! rief Fräulein Gröhl und zerknitterte eine eben empfangene Anzeige. In diesem Augenblicke trat Dr. Klausmann ein.",
                                    "Es ist am Abend des Festes Mariä Himmelfahrt. Die Haide liegt im Mondenglanze: in den dünenartigen, sandigen Hügeln bergen sich vorgeschichtliche Aschenurnen, die mächtigen Decksteine der Hünengräber aber wurden gesprengt und über die nahe holländische Grenze gefahren, zu Hafenbauten oder Seedeichbefestigungen."],
                                     trained_model, tokenizer)

print(generated_novellas)

print("finished!")