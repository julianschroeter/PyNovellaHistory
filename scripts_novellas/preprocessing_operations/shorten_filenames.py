import os
import re

from preprocessing.presetting import local_temp_directory, global_corpus_directory, set_DistReading_directory

system = "my_xps"

corpus_path = os.path.join(local_temp_directory(system), "episodes_5chunks")
corpus_path = global_corpus_directory(system)
new_dir = os.path.join(local_temp_directory(system), "episodes_5chunks_kurz")
new_dir = os.path.join(set_DistReading_directory(system),"novella_corpus_all_shortnames")

if not os.path.exists(new_dir):
    os.makedirs(new_dir)

for fn in os.listdir(corpus_path):

    basename, ext = os.path.splitext(fn)
    print(basename)
    doc_id = re.search("\d{5}-\d{2}", basename).group() if re.search("\d{5}-\d{2}", basename) else basename[:8]
   # chunk = re.search("\d{4}$", basename).group() if re.search("\d{4}$", basename) else basename[:-4]
    print(doc_id)
#    print(chunk)
    new_name = str(doc_id)
    print(new_name)

    text = open(os.path.join(corpus_path, fn), "r", encoding="utf8").read()
    with open(os.path.join(new_dir, new_name), "w") as f:
        f.write(text)
        f.close()
