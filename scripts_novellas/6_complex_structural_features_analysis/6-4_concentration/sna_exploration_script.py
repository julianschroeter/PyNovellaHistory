from collections import Counter
import re

all_character_occurences = ["König Julian", "Julian", "Karl", "König Julian", " König Karl", "Julian Schröter", "Karl Mayr", "Julian", "König Julian Schröter", "Karl", "König", "Schröter", "Mayr"]
counter_all_occurences = Counter(all_character_occurences)
dict_less_more_frequent_name_parts = {}

for character in all_character_occurences:
    # if name consists of two or more name parts (for example forname + surname), select that part which appears more often in the whole text
    name_parts = character.split(" ")
    if len(name_parts) > 1:
        new_counter_dict = {}
        for part in name_parts:
            print(part)
            new_counter_dict[part] = counter_all_occurences[part]

        more_frequent_name = max(new_counter_dict, key=new_counter_dict.get)
        less_frequent_name = re.sub(more_frequent_name, "", character)
        less_frequent_name = less_frequent_name.replace(" ", "")
        print("seltenerer Nagem", less_frequent_name)

        dict_less_more_frequent_name_parts[less_frequent_name] = more_frequent_name

print("seltenere Namensteile:", dict_less_more_frequent_name_parts.keys())
print("häufigere Namensteiel:", dict_less_more_frequent_name_parts.values())
print("ganzes dict", dict_less_more_frequent_name_parts)

new_all_character_occurences = []
for character in all_character_occurences:
    name_parts = character.split(" ")
    if len(name_parts) > 1:
        for part in name_parts:
            print(part)
            if part in dict_less_more_frequent_name_parts.values():
                new_all_character_occurences.append(part)


    elif len(name_parts) == 1:
        print("Das ist ein Name mit Länge 1", character)
        if character in " ".join(list(dict_less_more_frequent_name_parts.keys())):
            print("Dieser Name wird als der seltenere erkennt:", character)
            new_all_character_occurences.append(dict_less_more_frequent_name_parts[character])
        else:
            new_all_character_occurences.append(character)

print(new_all_character_occurences)