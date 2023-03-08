system = "wcph113"

from preprocessing.presetting import local_temp_directory
import os
import pandas as pd
from preprocessing.corpus import ChunkCorpus, generate_text_files
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
typical_novellas_ids = df_novellas[df_novellas["mean"] > 0.55].index.tolist()

print(typical_novellas_ids)

sentences_df = pd.read_csv(os.path.join(local_temp_directory(system), "novella_corpus_sentences.csv"), index_col=0)

print(sentences_df)

N_sent_samples = select_from_corpus_df(sentences_df,10, typical_novellas_ids, id_cat="doc_id")

print(N_sent_samples)

print(N_sent_samples.sent_string)