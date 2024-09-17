import requests
from bs4 import BeautifulSoup
from unidecode import unidecode
import pickle
import pandas as pd
import os

cache = True
url = "https://www.cesky-jazyk.cz"
if cache:
    import sys
    cache_file = "cesky-jazyk.cz.cache"
    sys.setrecursionlimit(100000)

def get_html(url, first_request=False):
    if os.path.exists(cache_file) and first_request:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        response = requests.get(url)
        html = response.content
        soup = BeautifulSoup(html, "html.parser")
        with open(cache_file, 'wb') as f:
            pickle.dump(soup, f)
        return soup

li_tags = get_html(url + "/slohovky/uvahy/", True).select("div.newstext.with div.over ul li.outside a")
links = []
for el in li_tags:
    if el.has_key("href"):
        links.append(el["href"])

titles = []
texts = []

n = len(links)
i = 0
for link in links:
    i += 1
    print(f"Status: {i}/{n}")
    soup = get_html(link)
    title = soup.select("h3.name")
    if len(title) > 0:
        title_text = title[0].text
    else:
        title_text = "None"
        print(f"Warning: title: {title_text}!")

    text = soup.select("html body div#main div#pozadi_pruh div#pozadi div#obsah div#main_part div#right_part div#middle_part div.cornerbox2 div.cornerboxinner2 div.newstext.with div#dbtext.over")
    for element in soup.select(".zdroj"):
        element.decompose()

    if len(text) > 0:
        text = text[0].text
    else:
        text = "None"
        print(f"Warning: text: {text}!")

    titles.append(title_text.strip())
    texts.append(text.strip())

data = pd.DataFrame({"titles": titles, "texts": texts, "links": links})
data.to_pickle("mydata.pkl")



#data = title + "\n" + text
