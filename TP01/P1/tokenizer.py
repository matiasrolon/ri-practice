import re
import unicodedata

def translate(string):
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
        ("ü", "u")
    )
    for a, b in replacements:
        string = string.replace(a, b).replace(a.upper(), b.upper())
    return string

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
