# Small example hot to convert german special characters from unicode to utf-8 and back to unicode
# http://www.utf8-zeichentabelle.de/unicode-utf8-table.pl?start=128&number=128&names=-&utf8=string-literal
#


import os
import re
from os.path import basename

def read_file(filepath):
   text = open(filepath, "r", encoding="utf8").read()
   return text

umlaut_translation_table = {
    "Ä" : "Ae",
    "ä" : "ae",
    "Ö" : "Oe",
    "ö" : "oe",
    "Ü" : "Ue",
    "ü" : "ue",

}


stoerzeichen_table = {
    "^" : "",
    "%" : "",
    "&" : "",
    "♦" : "",
    "#" : "",
    "•" : "",
    "■" : "",
    "§" : "",
    "|" : "",
    "„" : '''"''',
}

def translate_german_umlaute_string(string):
    'ersetze umlaute gemäß umlaut_translation_table'
    return(string.translate(str.maketrans(umlaut_translation_table)))


def translate_german_sz_to_ss_string(string):
    'ersetze ß durch ss'
    return(string.translate(str.maketrans({'ß' : 'ss'})))

def remove_stoerzeichen_string(string):
    'ersetze bzw. entferne Stoerzeichen gemaeß stoerzeichen_table'
    return(string.translate(str.maketrans(stoerzeichen_table)))


# für neueren workflow mit bearbeitetem Seitenumbruch: " \umbruch + Seitenzahl und oder Titel/Autor-Angabe"
# für Taschenbücher wie "Urania", "Rheinisches TB", "Iris" "Aglaja" usw
# ersetze Umlaute und special characters mit replace_german_umlaute
# Entferne "whitespace . whitespace", "Leerzeilen Seitenzahl Leerzeilen", "Silbentrennung", "einfache Zeilenumbrüche"


def remove_breaks_pagecounts(input_dir, output_dir):
    'removes hyphenation and linebreaks for files in input_dir, new files are written to output_dir'
    corpus_path = os.path.join(input_dir)
    corpus_filenames = [os.path.join(corpus_path, fn) for fn in sorted(os.listdir(corpus_path))]
    for filename in corpus_filenames:
        text = read_file(filename)
        # entferne die Seitenumbrüche mit anschließender Seitenangabe und evtl. Titel oder Autorangabe
        outfile_text = re.sub("\u000C\d{0,3}\D{0,20}\d{0,3}\n", "", text)
        # entferne Silbentrennung
        outfile_text = outfile_text.replace("-\n", "")
        # negative lookahaed: entferne Zeilenumbrüche, die keinen Absatz markieren
        outfile_text = re.sub("\n(?!\n)", " ", outfile_text)
        # entferne den zuvor entsandenen whitespace
        outfile_text = re.sub("\n\s", "\n", outfile_text)
        # entferne alle übrigen Seitenumbrüche
        outfile_text = re.sub("\u000C", "", outfile_text)
        # entferne Seitenangaben in eckigen Klammern
        outfile_text = re.sub("\[\d{2,3}\]", "", outfile_text)
        outfile_name = open(os.path.join(output_dir, basename(filename)), 'w')
        outfile_name.write(outfile_text)
        outfile_name.close()
        pass




def remove_stoerzeichen(input_dir, output_dir):
    corpus_path = os.path.join(input_dir)
    corpus_filenames = [os.path.join(corpus_path, fn) for fn in sorted(os.listdir(corpus_path))]
    for filename in corpus_filenames:
        text = read_file(filename)
        outfile_text = remove_stoerzeichen_string(text)
        outfile_name = open(os.path.join(output_dir, basename(filename)), 'w')
        outfile_name.write(outfile_text)
        outfile_name.close()
        pass


def normalize_rembreaks_subdir(parent_dir, output_parent_dir):
    corpus_path = os.path.join(parent_dir)
    subdir_list = (os.listdir(corpus_path))
    for subdir in subdir_list:
        subdir_lastsegment = os.path.basename(os.path.normpath(subdir))
        if not os.path.exists(os.path.join(output_parent_dir, subdir_lastsegment)):
            os.mkdir(os.path.join(output_parent_dir, subdir_lastsegment))
        else:
            pass
        subdir_path = os.path.join(parent_dir, subdir)
        subdir_filenames = [os.path.join(subdir_path, fn) for fn in sorted(os.listdir(subdir_path))]
        for filename in subdir_filenames:
            text = read_file(filename)

            outfile_text = re.sub("\u000C\d{0,3}\D{0,20}\d{0,3}\n", "",
                                  outfile_text)  # entferne die Seitenumbrüche mit anschließender Seitenangabe und evtl. Titel oder Autorangabe
            outfile_text = outfile_text.replace("-\n", "")  # entferne Silbentrennung
            outfile_text = re.sub("\n(?!\n)", " ",
                                  outfile_text)  # negative lookahaed: entferne Zeilenumbrüche, die keinen Absatz markieren
            outfile_text = re.sub("\n\s", "\n", outfile_text)  # entferne den zuvor entstandenen whitespace
            outfile_text = outfile_text.replace(",,", '''"''')
            outfile_text = re.sub("\u000C", "", outfile_text)  # entferne alle übrigen Seitenumbrüche
            outfile_text = re.sub("\[\d{2,3}\]", "", outfile_text)  # entferne Seitenangaben in eckigen Klammern
            outfile_name = open(os.path.join(output_parent_dir, subdir_lastsegment, basename(filename)), 'w')
            outfile_name.write(outfile_text)
            outfile_name.close()




def normalisieren(input_dir, output_dir):
    "Angleichung an heutige Schreibweisen, insb. th->t, ey->ei, c->k usw."
    corpus_path = os.path.join(input_dir)
    corpus_filenames = [os.path.join(corpus_path, fn) for fn in sorted(os.listdir(corpus_path))]
    for filename in corpus_filenames:
        text = read_file(filename)
        text = text.replace('Besorgniss', 'Besorgnis')
        text = text.replace('besorgniss', 'besorgnis')
        text = text.replace('Capelle', 'Kapelle')
        text = text.replace('Uebermuth', 'Uebermut')
        text = text.replace('uebermueth', 'uebermuet')
        text = text.replace('Urtheil', 'Urteil')
        text = text.replace('urtheil', 'urteil')
        text = text.replace('Athem', 'Atem')
        text = text.replace('athem', 'atem')
        text = text.replace('Theil', 'Teil')
        text = text.replace('theil', 'teil')
        text = text.replace('That', 'Tat')
        text = text.replace('that', 'tat')
        text = text.replace('gethan', 'getan')
        text = text.replace("""'s""", 's')
        text = text.replace("""’s""", 's')
        text = text.replace(""" 'ge""", 'ge')
        text = text.replace("Vadder", 'Vater')
        text = text.replace("gebuerge", 'gebirge')
        text = text.replace("Gebuerge", 'Gebirge')
        text = text.replace("ey", "ei")
        text = text.replace("koeniginn", "koenigin")
        text = text.replace("Koeniginn", "Koenigin")
        text = text.replace("Commandant", "Kommandant")
        text = text.replace("commandant", "kommandant")
        text = text.replace("October", "Oktober")
        text = text.replace("october", "oktober")
        text = text.replace("Geheimniß", "Geheimnis")
        text = text.replace("geheimniß", "geheimnis")
        text = text.replace("Geheimniss", "Geheimnis")
        text = text.replace("geheimniss", "geheimnis")
        outfile_name = open(os.path.join(output_dir, basename(filename)), 'w')
        outfile_name.write(text)
        outfile_name.close()

