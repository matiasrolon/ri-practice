import re
import unicodedata
from nltk.stem import SnowballStemmer

_stemmer = SnowballStemmer("spanish")

def translate(string):
    """
    Convierte caracteres acentuados a su forma no acentuada.
    """
    return (
        unicodedata.normalize("NFD", string)
        .encode("ascii", "ignore")
        .decode("ascii")
    )

def tokenize(text):
    """
    Tokeniza el texto extrayendo solo palabras comunes (letras),
    igual que P2, pero aplicando stemming con SnowballStemmer (español).
    """
    text = text.lower()
    text = translate(text)
    word_re = re.compile(r"[a-z]+(?:[-'][a-z]+)*")
    tokens = word_re.findall(text)
    return [_stemmer.stem(tok) for tok in tokens]
