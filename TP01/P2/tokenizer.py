import re
import unicodedata

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
    Tokeniza el texto extrayendo solo palabras comunes (letras).
    """
    text = text.lower()
    text = translate(text)
    word_re = re.compile(r"[a-z]+(?:[-'][a-z]+)*")
    return word_re.findall(text)