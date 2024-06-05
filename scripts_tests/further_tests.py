from datetime import datetime
from bs4 import BeautifulSoup
import urllib
current = datetime.now()
print(current)

with urllib.request.urlopen("http://www.stern.de/") as output:
    sourcecode = output.read().decode("utf-8")

html_soup = BeautifulSoup(sourcecode, "html.parser")
#print(html_soup)

a_tags = html_soup.find_all("a")
for a_tag in a_tags:
    try:
        a_tag = a_tag.attrs["href"]
        print(a_tag)
    except:
        print("Konnte href nicht holen")