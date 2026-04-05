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

def tokenize(string):
    """
    Realiza la tokenización de una cadena de texto, 
    eliminando caracteres especiales, pasando a minúsculas,
    y removiendo acentos.
    """

    string = re.sub(r'[^\w\s]|_', '', string)
    string = string.lower()         # pasaje a miscula
    string = translate(string)      # quita acentos
    return string.split()           # detecta tokens
