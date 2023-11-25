The PyNovellaHistory repository contains all python code for the text analysis within the novella history project (Habilitationsprojekt). 

A (German) project description can be found here: https://www.germanistik.uni-muenchen.de/personal/ndl/professoren/schroeter/projekte/index.html, Parts of the project were conducted within a Walter-Benjamin Fellowship, funded by the German Research Foundation (DFG): https://gepris.dfg.de/gepris/projekt/449668519?language=en

For replicating the analysis of the paper J. Schröter: "Machine Learning as a Measure of the Conceptual Looseness of Disordered Genres: Studies on German Novellen." In: Christof Schöch, Robert Hesselbach (eds.): Digital Stylistics in Romance Studies and Beyond, Heidelberg University Publishing 2023, use the scripts in the "pca_distance_plots" and "perspectival_classificaton_genres" folders. Note that the analyses of that paper were conducted in 2020. Since then, the corpus has considerably been extended so that the original results of the paper cannot be replicated exactly.

The python (version 3) code is organized as modules. The basic structure of the modules starts with objects as text representations (Text class in Preprocessing.Text). Based on text representations, corpus representations objects are generated as document features matrices (or child classes such as document term matrices). The parent class is FeatureDocMatrix in the Preprocessing.Corpus module.

The scripts to reproduce the analyses of the project can be found in the scripts_novellas folder.

The corpus representation object allows integrating meta data from metadata csv files.

POS-Tagging and NE-Recognition is based on the spacy library, and it uses a custom trained large german language model.

In the Preprocessing.Presetting module, filepath to the corpus files can be customized in a systematic way. It features a path to the corpus txt files (whole corpus and a small test corpus), a path to the customized language model, a path to corpus-representation files, a path to metadata files, and a path to output-files.

SNA (social network analyis tools) are featured in the preprocessing area: In the first step, references on fictional characters are extracted based on explicit rules and NER. Secondly, centrality measures for character networks are stored as text representations and can be included to corpus representations

A set of further modules includes functions for different types of text analysis:
- postprocessing for Topic Modeling based on mallet,
- PCA for different feature representations: DTM
- Supervised Learning (classification tasks, mostly organized as "perspective modeling", inspired by Ted Underwood, Distant Horizons 2019)


The folder Scripts contains all python scripts that execute text analyses based on the the above mentioned modules.


The following libraries are required:
- numpy
- pandas
- spacy
- sklearn
- networkx
- matplotlib

The following built-in modules are used: os, copy, collections, pickle, re, string, itertools,
