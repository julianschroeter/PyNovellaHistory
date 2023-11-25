import os
import re

from preprocessing.presetting import local_temp_directory

system = "my_xps"

corpus_path = os.path.join(local_temp_directory(system), "episodes_5chunks")
new_dir = os.path.join(local_temp_directory(system), "episodes_5chunks_kurz")
for fn in os.listdir(corpus_path):

    basename, ext = os.path.splitext(fn)
    print(basename)
    doc_id = re.search("\d{5}-\d{2}", basename).group() if re.search("\d{5}-\d{2}", basename) else basename[:8]
    chunk = re.search("\d{4}$", basename).group() if re.search("\d{4}$", basename) else basename[:-4]
    print(doc_id)
    print(chunk)
    new_name = str(doc_id+ "_"+ chunk)
    print(new_name)

    text = open(os.path.join(corpus_path, fn), "r", encoding="utf8").read()
    with open(os.path.join(new_dir, new_name), "w") as f:
        f.write(text)
        f.close()
