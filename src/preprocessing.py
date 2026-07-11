import re
import emoji
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords',quiet=True)
def text_preprocessing(text,remove_stopwords=False):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+","",text)
    text = re.sub(r'&[a-z]+;|&#\d+;', '', text) 
    text = re.sub(r'<[^>]+>', '', text)
    text = emoji.replace_emoji(text,replace="")
    special_character_pattern = r'[^a-zA-Z0-9\s]+'
    text = re.sub(special_character_pattern,"",text)
    text = re.sub(r"\s+"," ",text).strip()
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        words = text.split(" ")
        words = [word for word in words if word not in stop_words]
        text = " ".join(words)
    
    return text