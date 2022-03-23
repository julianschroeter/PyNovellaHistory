
# coding: utf8
"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.
For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities
Compatible with: spaCy v2.0.0+
Last tested with: v2.2.4
"""
from __future__ import unicode_literals, print_function

import plac
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# training data
TRAIN_DATA = [
    # von Julian:
    ("Andreas entgegnete: so wäre leicht Rath zu finden", {"entities": [(0, 7, "PER")]}),
    ("Heinrich sagte: ein Sinnbild der Ewigkeit, wie Andre sagen", {"entities": [(0, 8, "PER")]}),
    ("Clara sagte: Kann ein Tagebuch etwas Andres als Monologe enthalten?", {"entities": [(0, 5, "PER")]}),
    ("Ja, Christine ist doch mehr werth", {"entities": [(4, 13, "PER")]}),
    ("Niemand kam so häufig auf die Burg als Philipp Walther", {"entities": [(39, 54, "PER")]}),
    ("Als das Abendessen abgetragen war, und sich die Knechte wieder entfernt hatten, nahm Eckbert die Hand Walthers und sagte:", {"entities": [(85,92, "PER"), (102, 110, "PER")]}),
    ("er durfte seiner Frau wegen diese Tochter nicht bei sich erziehn lassen", {"entities": []}),
    ("O wie mich freut Waldeinsamkeit", {"entities": []}),
    ("Ach einzge Freud Waldeinsamkeit!", {"entities": []}),
    ("wenn ich es beschreiben soll, so war es fast, als wenn Waldhorn und Schalmeie ganz in der Ferne durcheinanderspielen", {"entities": []}),
    ("Die Verlobung von St. Domingo", {"entities": [(18, 29, "LOC")]}),
    ("Ihr wäret sammt der lieblichen jungen Mestize, die mir das Haus aufmachte, mit uns Europäern in Einer Verdammniß?", {"entities": []}),
    ("meint Ihr, daß das kleine Eigenthum, das wir uns in mühseligen und jammervollen Jahren durch die Arbeit unserer Hände erworben haben, dies grimmige, aus der Hölle stammende Räubergesindel nicht reizt?", {"entities": []}),
    ("Sie sagte, daß sie vor fünfzehn Jahren auf einer Reise, welche sie mit seiner Frau nach Europa gemacht hätte, in Paris von ihr empfangen und geboren worden wäre.", {"entities": [(88, 94, "LOC"), (113, 118, "LOC")]}),
    ("Ich hatte sie in dieser Stadt, wo ihr Vater Kaufmann war, kurz vor dem Ausbruch der Revolution kennen gelernt", {"entities": []}),
    ("ein fürchterlicher alter Neger, Namens Congo Hoango", {"entities": [(39, 51, "PER")]}),
    ("eine alte Mulattin, Namens Babekan", {"entities": [(27, 34, "PER")]}),
    # von Johannes:
("Ich hatte meinen Bedienten, von dem ich auf eine grobe Weise betrogen worden, am Tage vor meiner Abreise weggejagt, und war also in diesem Augenblick, in buchstäblichem Verstände, Niemands Herr noch Knecht.",{"entities":[]}),
("Der Prinz hielt vor dem Balcon und sprach einige Worte zu Frau v. Normand, die sich dabei ein paar Mal ungeduldig nach der offenen Balcons-Thüre umsah.", {"entities":[(58,73,"PER")]}),
("Trotz allem, was ich Böses von ihm gehört und selbst erfahren hatte, konnte ich einem mit dem Tode Ringenden die Gewährung eines so einfachen Verlangens nicht verweigern.", {"entities":[]}),
("Aber dieser kluge Plan ward durch den unzeitigen Tod des edlen jungen Mannes vereitelt, und Frau von M. sah sich genöthigt, sehr zweifelhafte Ansprüche gegen eine aufgebrachte und mächtige Familie in einem fremden Lande geltend zu machen.", {"entities":[(92,103,"PER")]}),
("Als ich von meinem Ausflug in's Gebirge, schon ziemlich spät, heimkehrte, sah ich auf der Straße, die nach D. und L. führt, eine Kutsche mit Vieren mir entgegen kommen und rasch an mir vorbeifahren.", {"entities":[(108,109,"LOC"),(114,115,"LOC")]}),
("Graf Arthur von S., ursprünglich einer englischen Familie entsprossen, war in Polen geboren, und vereinte Alles, was diese Nation auszeichnet, mit Tapferkeit, Freisinn und Vaterlandsliebe, einen Hang zur Schwermuth, und einen Ernst, welchen wahrscheinlich seine Vorfahren ihm als Erbstück ihres neblichten Clima's hinterlassen hatten.", {"entities": [(0, 18, "PER"), (78, 83, "LOC")]}),
("Stiller Wohnsitz meiner Väter!", {"entities":[]}),
("Lazienki bedeutet Bäder.", {"entities":[(0, 8, "LOC")]}),
("An einem jener Herbstabende, wo die Sonne nur hinter einem Nebelschleier lächelt und untergeht, hatte er sich in dem Wäldchen von Bilany **) verspätet.", {"entities":[(130, 136, "LOC")]}),
("Es ist die Gräfin Ludmilla von O...", {"entities":[(11, 32, "PER")]}),
("Am Morgen hatte Graf S. die Residenz verlassen, und Ludmillen ward folgender Brief eingehändigt, welcher in polnischer Sprache geschrieben, die sich etwas der des Orients in ihrem Bilderschmucke nähert, vielleicht schwer zu übersetzen seyn wird.", {"entities":[(16, 23, "PER")]}),
("Wie schön bist du, und vergessen war der unberufene Warner.", {"entities":[]}),
("Während dem setzte Graf S. seine Reise schnell, als ob irgend eine Gefahr ihm auf der Ferse folgte, nach der Heimat fort.", {"entities":[(19, 26, "PER")]}),
("Die Starostin lud ihn ein, längere Zeit bei ihr zu verweilen, und kam damit seinen Wünschen zuvor, dennoch zweifelte er, ob er ihre Güte benutzen sollte, wenn er bedachte, welch ein trauriger Gesellschafter er sey.", {"entities":[(4, 13, "PER")]}),
("Ward sie in solchen Empfindungen durch Arthur überrascht, so umspielte ein sanftes Lächeln ihren schönen Mund, ihre Stimme war weicher, und ohne Worte berechtigte sie ihn zu dem Glauben, es sey ihm gelungen, ihr Herz zu rühren.", {"entities":[(39, 45, "PER")]}),
("Meine Tochter! sprach sie, mit Wehmuth sehe ich dich mich verlassen, mit Wehmuth; aber ohne Furcht, denn du hast nun kennen gelernt, worin das wahre Glück besteht; nicht in einem geräuschvollen zerstreuten Leben, unter einer bewundernden Menge, sondern in dem Beifall eines Herzens, das man ganz ausfüllt, und das alle Schmeicheleien der Welt aufwiegt.", {"entities":[]}),
("Einige Augenblicke blieb er unbeweglich stehen, dem Wagen nachstarrend, der ihm Freude, Ruhe und Hoffnung entführte, doch bald war er seinen Blicken entschwunden, das Geräusch seiner Räder verhallt, und Todtenstille war eingetreten.", {"entities":[]}),
("Todtenblässe bedeckt sein Gesicht, mit einem Schrei, wie die Posaune des Weltgerichts, ruft er: Ludmilla! und stürzt besinnungslos zu Boden.", {"entities":[(96, 105, "PER")]}),
("Graf Arthur von S... hatte in seiner noch übrigen Lebenszeit öftere Anfälle von Geistesabwesenheit, und fiel endlich, ein williges Opfer, in den Freiheitskriegen seines Landes.", {"entities":[(0, 17, "PER")]}),
("In einer jener langen finstern Gassen des alten Dresdens, die so eng sind, daß fast eine Schwalbe beim Durchflug mit der Spitze ihrer Flügel die gegenüberstehenden schwarzen Steinwände zugleich berühren muß, lebte zur Zeit des Churfürsten Christian II. zu Anfang des siebzehnten Jahrhunderts ein armer Goldschmiedmeister mit Namen Tobias Rosen.", {"entities":[(48,56,"LOC"),(243,258,"PER"), (337,349,"PER")]}),
("Er mußte endlich, da sie ihn zu erdrücken drohten, sich selber kräftig aufraffen; er that es und stand auf, um in die Kammer seines Weibes zu gehen, das, wie er wußte, eben beschäftigt war, den Sonntagsputz anzulegen, um eine Gevatterin zu besuchen, die vor dem Thor wohnte und heute ein kleines Fest in ihrem Garten feierte.", {"entities":[]}),
("Auf diese Weise lebte der Kranke wol zwei Wochen in Tobias' Hause.", {"entities":[(52,58,"PER")]}),
("Weiht mich nicht zwecklos in Euere Geheimnisse ein, bemerkte der Goldschmiedmeister ernst.", {"entities":[]}),
("Nein, entgegnete der Wirth.", {"entities":[]}),
("Er führt die Holde ihrem Vater zurück und erhalt sie von diesem zur Gemahlin zum Dank für seine kühne Ritterthat.", {"entities":[]}),
("Viele hielten sie für ein Product des menschlichen Fleißes, wie die Katakomben in Aegypten und Italien, Andere für ein Werk der Natur, wie die Höhle auf Antipares, und die Dichter meinten gar, sie sey der Eingang in die Unterwelt, durch welche Hercules dem König Admet seine theure Gattinn heraufgeholt, die er — zur Verwunderung aller Ehemänner — vom Orcus zurück verlangte.", {"entities":[(153,162,"LOC"), (245,252,"PER"), (263,269,"PER"), (353,358,"LOC")]}),
("Aber bald entdeckte ich, daß es eine Heerde Hornvieh war.", {"entities":[]}),
("Da diese Meinung sich sogar bis in den Mittelpunkt der Erde verbreitet hat, so ist es kein Wunder, daß auch unser Adel von besserem Blute zu seyn glaubt, als wir Bürgerliche; wie auch die Idee von Stammbäumen wohl von dem Planeten Nazar zu uns gekommen seyn mag.", {"entities":[(231,236,"LOC")]}),
("In Coklecu regieren die Weiber ausschließend die Männer verrichten die gemeineren Arbeiten, was beiden Theilen recht wohl zu behagen scheint, denn die Männer sind träge und die Weiber verliebt.", {"entities":[(3,10,"LOC")]}),
("Ferner Pastor Tam's Beichtreden, welche das rasche Blut verdünnen, und zugleich den Schlaf befördern etc. etc.", {"entities":[(14,19,"PER")]}),
("Die Frau Ministerinn ließ mich öfters, jedoch sonderbar genug; immer nur wenn ihr Mann nicht zu Hause war, zu sich rufen.", {"entities":[]}),
("Was sind dagegen Ruhm und all die Possen, Nach welchen man, zu Fuß und in Carrossen, Sich müde ringt, doch nicht den Hunger stillt?", {"entities":[]}),
("Im ersten Augenblick sah der Arzt bald den entschlafenen Eber, bald die erstarrte Haushälterinn, bald den von Gewissensangst gefolterten Bartkünstler an.", {"entities":[]}),
("Niemand blieb, als John Pfeiffer, und dieser erklärte endlich, daß er es unter der Bedingung übernehmen wolle, dem Herrn Horpiz, wenn er wiederkomme, den Werth dafür zu bezahlen, außer diesem Fall aber, weil es doch immer eine gewagte Sache bleibe, sich mit dem Teufel in einen Proceß einzulassen, und weil er bloß den Zweck habe, die hochweisen Herren aus einer Velegenheit zu ziehen, müsse es ihm, zur Belohnung seines Muths und seines Patriotismus unentgeldlich verbleiben.", {"entities":[(19,32,"PER"),(115,127,"PER")]}),
("Um dem Fremden aber doch einen Nachgenuß einer so sehenswürdigen Feierlichkeit zu verschaffen, führte er ihn zu dem vor dem Thor gelegenen Galgen, und staunte nicht wenig, als er von ferne den flatternden Schlafrock erblickte, der mit dem enganliegenden Armensünderkleid, von grauem Zwillich mit schwarzer Einfassung, auch nicht die mindeste Aehnlichkeit hatte.", {"entities":[]}),
("Es freut mich aber recht, lieber Carl, du bist ihm jedenfalls einen Besuch schuldig gewesen.", {"entities":[(33,37,"PER")]}),
("„Durchaus ehrlich,“ antwortete jener: „der Vergleich wird von dem Vater auf Rechnung der möglichen Mariage, also auf meine Rechnung eingegangen; wird später nichts aus solcher, so kommt das auf meiner Tochter Rechnung, und jedenfalls hat dann Herr von Espen junior Schuld, wenn er nicht im Stande gewesen ist, das Herz des Mädchens zu gewinnen, das auch zur Mariage mitgehört.", {"entities":[(243,263,"PER")]}),
# von Theresa
("Leuchtkäfer schwebten, mit stillem, grünlichem Lichte durch das Dunkel, kreisten vor den Frauen herum, und Gertrud schien ihr Spiel mit bedeutendem Ernst zu betrachten.", {"entities": [(107, 114, "PER")]}),
("nur fordere sie sieben Haare von Gertrudens Haupt.", {"entities": [(33, 43, "PER")]}),
("Übrigens kann dann nichts geschehen, und wie ihr mich da seht, — frisch und gesund, und so eine gute Christin, als ihr seyd, — bin ich schon mehr als einmal Zeugin solcher wunderbaren Dinge gewesen, und habe Sachen gesehen, — Sachen , — deren Wiederholung euch ein Mährchen bedünken würde.", {"entities": []}),
("Das thaten Sie wirklich, Herr Samuel Brink?",{"entities": [(30, 42, "PER")]}),
("Gretchen sey sehr bekümmert, erzählte er, daß ich nicht zum Nachtessen käme, und Max hab' es sich gleichfalls verbothen.",{"entities": [(0, 8, "PER"),(81,84, "PER")]}),
("Unwillkührlich folgte ich der anziehenden Erscheinung, die, durch das Thor an meinem Wagen vorbey eilend, meinen Blicken bald entschwand.",{"entities": []}),
("Der Ernst desselben schien sich späterhin dadurch noch mehr zu bewahren, daß jetzt auf Einmahl im Wochenblatte die Anzeige erschien, in der Wohnung der Mamsell Emerentia Kasimir stünden zwey Quartiere, jedes für einen einzelnen, stillen Herrn, sogleich zu vermiethen.",{"entities": [(160, 177, "PER")]}),
("Lieber Kneller — begann Herr Schilf — wir sind die ältesten Freunde, daher muß ich Dir nothwendig von meiner neuesten Speculation zuerst sagen.",{"entities": [(7, 14, "PER"),(29, 35, "PER")]}),
("Der Oberbürgermeister kam heute, wie gewöhnlich, um die Mittagsstunde aus der Sitzung zurück, und erfuhr von seinem alten Friedrich, es sey ein junges Mädchen, Luise Müller mit Namen , hier gewesen, den Herrn zu sprechen;", {"entities":[(122, 131, "PER"),(160,172, "PER")]}),
("Sie heißen Maria Elisabeth Müller?", {"entities":[(11, 33, "PER")]}),
("Dabey blieb es;", {"entities":[]}),
("auch die Uebrigen von Lorenzo von Bologna waren nie so zu sehn als jetzt in der Lichtglut der Kerzen!", {"entities":[(22, 41, "PER")]}),
("In der Cathedrale von Messina in dem linken Seitenschiff ist freylich die großartigste, eine ganze Halle füllende Madonna — aber da waren wir leider jetzt nicht;", {"entities":[(8, 29, "LOC")]}),
("Ostern war spät im Jahre gefallen, und ob das Frühjahr gleich hier nicht viel schöner ist als das ganze Jahr, so umfing uns Wandernde doch Frühlingsfrische, begegneten wir neu und kräftig grünende Bäume, und ein reiner blauer Himmel umwölbte uns.", {"entities":[]}),
("Endlich gingen wir wieder nach Velletri.", {"entities":[(32, 39, "LOC")]}),
("König Friedrich August der Erste von Polen war gestorben, und die Unruhen, welche einen großen Theil seiner Regierung getrübt, und nur gegen das Ende derselben sich einigermaßen gestillt hatten, schienen nach seinem Tode wieder anzufangen.", {"entities":[(0, 42, "PER")]}),
("Ein Theil, unter russischem Einfluß, suchte dem Sohne des eben verstorbenen August die Thronfolge zu sichern, während der Andere, von französischem Interesse beseelt, den edlen Stanislaus Leczinsky, den Schwiegervater König Ludwig des Fünfzehnten, wieder auf den Thron erheben wollte, den er schon früher durch Karl des Zwölften siegreiche Waffen eine Weile gegen Friedrich August behauptet, und nur nach dem Tode des heldenmüthigen Schwedenkönigs seinem Nebenbuhler hatte überlassen müssen.", {"entities":[(76, 82, "PER"),(177, 197, "PER"),(218, 246, "PER"),(311, 328, "PER"),(364, 380, "PER")]}),
("Da nun dieser Zwiespalt sie quälte, da seine Vermählung herannahte, und er schon gegen den Chevalier den Wunsch geäußert hatte, seine junge Gemahlin bei Elliska aufzuführen , und diese um ihre Freundschaft für Fräulein Dammartin zu bitten, so faßte Elliska den Entschluß, unter irgendeinem anständigen Vorwand Lüneville zu verlassen, und nicht eher wieder dahin zurück zu kehren, bis der Herzog mit seiner Gemahlin nach Paris gegangen seyn würde.", {"entities":[(153, 160, "PER"),(249, 228, "PER"),(249, 256, "PER"),(310, 319, "LOC"),(420, 425, "LOC")]}),
("Wie wenn Einer in einem säulengetragenen Gemache stünde und scherzend sagte: »Jetzt will ich diese Säule da umreißen, damit der ganze Plunder zusammenstürze,« und dann die Säule zum Scherze faßte und rüttelte, und sie bräche wirklich ein, und die Decke stürzte mit schmetterndem Geprassel nieder: so erging es mir.", {"entities": []}),
("Der junge Mann bekam einen Ruf nach Grätz.", {"entities": [(36, 41, "LOC")]}),
("Sesa kam über den Gang.", {"entities": [(0, 4, "PER")]}),
("Sigismunda und ihr Ritter sezten sich mit ihrem hohen Gaste, und während Herold den Grafen Ravensberg aufsuchte, ging das Fräulein an den Putztisch, den sie noch nie so lange vergessen hatte.", {"entities": [(0, 10, "PER"),(73, 79, "PER"),(84, 101, "PER")]}),
("»Dann wird Herold von Hochstaden Euch geleiten« sagte Sigismunda, »bis Ihr sicher in der Hofburg des Bischofs angelangt seyd.", {"entities": [(11, 32, "PER"),(54, 64, "PER")]}),
("Stolz und Hoffarth veröden die Brust, wo sie wohnen, und eitle Gefallsucht ist ein Unkraut, neben welchen die Liebe nicht blühet.", {"entities": []}),
("Auch eine treue Dienerin, Gisela von Tresbach, fand sich in Bamberg zu ihr, eine Frau, die Muttergefühle für die junge Fürstin hegte, und mit lebhaftem Schmerz, das allmählige Absterben alles Irdischen, das Hinwelken der Jugendblüte, und die zunehmende Strenge gegen sich selbst bemerkte, die Elisabeth, wenige Jahre später, zur Heiligen erhoben.", {"entities": [(26, 45, "PER"),(60, 67, "LOC"),(293, 302, "PER")]}),
("Nataliens Schwermuth war nun groeßer als vorher geworden;", {"entities": [(0, 9, "PER")]}),
("Schuldlos bin ich, das schwoer' ich dir, und bald wird mit des Himmels Huelfe die Wahrheit siegen, die meine Unschuld an das Licht der Sonne stellt.", {"entities":[]}),
("Zur Zeit, als die Beherrscher Russlands ihren Sitz noch in Moskau hatten, und der entnervende Geist der Neuerungen seinen Thron noch nicht in dieser Hauptstadt aufgeschlagen hatte, lebte daselbst Matwey Andrejew, einer der angesehensten Bojaren.", {"entities": [(30, 39, "LOC"),(59, 65, "LOC"),(196, 211, "PER")]}),
("Bekuemmert sah Matwey Natalien scheiden, und ahnte nicht, dass er morgen auf lange, vielleicht auf immer, ein kinderloser Vater seyn werde.", {"entities": [(15, 21, "PER"),(22, 30, "PER")]}),
("Herold von Hochstaden, Sigismunda lebt, Ihr findet sie in dem Schutze der heiligen Elisabeth, an ihrem Grabe wieder, und findet sie, einer standhaften Liebe würdig: Gesegnet sey diese Nacht!", {"entities": [(0, 21, "PER"),(23, 33, "PER"),(83, 92, "PER")]}),
]

@plac.annotations(
    model=("de_core_news_lg", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model="de_core_news_lg", output_dir="/mnt/data/users/schroeter/CLS_temp/language_models", n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            print(ent)
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == "__main__":
    plac.call(main)