# Anleitung zur Verwendung des Repositoriums für die Replikation der Ergebnisse der Habiliationsschrift Julian Schröter: Gattung – Medium – Praxis (2024)

Dieses Repositorium versteht sich als eigenständige und modular aufgebaute Bibliothek zur computergestützten Analyse literarischer Texte, zugleich aber ist sie die programmtechnische Grundlage oder obengenannten Arbeit. 
Als solche dient das Repositorium dazu, sämtliche der in der Arbeit vollzogenen Analysen zu wiederholen, die Methoden weiterzuverwenden und, im besten Fall, weiter zu entwickeln.

Das Repositorium besteht aus zwei Teilen:

I) Den Modulen mit umfangreichen Klassen und Funktionen zur Textanalyse, sowie zum Pre- und Postprocessing. (zur Architektur siehe die Datei README.md

II) Den Skripten zur konkreten Durchführung der Analysen

Zu Teil II) Alle relevanten Skripte befinden sich im Ordner scripts_novellas. Darin ist die Kapitelstruktur des Buchs abgebildet. Im Buch wird in jedem Kapitel auf einige der relevanten Skripte verwiesen.

In Teil (I) sind, mit Blick auf die Replikation folgende Module besonders wichtig:

Das modul preprocessing.presetting enthält Funktionen, die für unterschiedliche Systeme systematisch auf Verzeichnisse zugreifen. Diese Funktionen sollten an die jeweiligen lokalen Gegebenheiten angepasst werden. 
Am einfachsten ist es, sich einen Zugang zu einem Hochleistungsrechner mit dem Namen "wcph113 am Lehrstuhl für Computerphilologie" der Universität Würzburg zu besorgen, auf dessen Verzeichnisstruktur die Skripte bereits zugeschnitten sind.

Wichtig sind folgende Funktionen und Verzeichnisse:

Die Funktion: global_corpus_directory(system_name) Die auf die Volltexte (plain text) des Projektkorpus verweist.

Die Funktion global_corpus_representation_directory(system_name), die auf das Verzeichnis verweist, in dem die Korpora hinsichtlich der extrahierten Features, in der Regel als Document-Feature-Matrizen (z.B. als Document-Term-Matrizen) repräsentiert und als tabellarische Dateien (.csv) gespeichert sind.

Die Funktion local_temp_directory(system_name), die auf das Verzeichnis verweist, das temporär für alle zwischengespeicherten Daten, Analyseergebnisse (hier auch für Abbildungen und tabellarisch gespeicherte Resultate) genutzt werden kann. Dieses Verzeichnis war im Verlauf der Projektarbeit der Sammelbehälter für fast alles


vocab_lists_dicts_directory(system_name) verweist auf ein Verzeichnis, in dem die für die Analysen  relevanten Listen, dictionaries, Wörterbücher etc. gespeichert sind.

Die Funktion conll_base_directory verweist auf das Verzeichnis, in dem die extrahierten Features (POS-Tags, Lemmata, NER, Koreferenzauflösung, Redewiedergabe) als Output der NLP-Pipeline LLPro (Ehrmanntraut/Jannidis/Konle 2023) gespeichert sind.

Für die grundlegende Architektur der KLassen und Funktionen ist wichtig, von zwei zentralen Klassen auszugehen: 

a) Der Klasse Text im Modul preprocessing.text, die Methoden bereitstellt, mit denen Features aus Textdateien extrahiert werden können. Unterklassen können auch komplexe Features extrahieren. 
b) Die Klasse DocFeatureMatrix im Modul preprocessing.corpus, die Methoden bereitstellt, um (mit der Text-Klasse) die Features aller (oder mehrerer) Texte  eines Korpus (oder Stichprobe) in einer Document-Feature-Matrix abzubilden. Die wichtigste Unterklasse ist DTM, die eine Document-Term-Matrix mit einer hohen Flexibilität zu Formen der Normalisierung und Featureselektion erzeugt. 
Weitere Unterklassen erstellen Repräsentationen komplexer Features auf Korpusebene. Dazu gehören: DocThemesMatrix, DocNetworkFeatureMatrix, etc.


