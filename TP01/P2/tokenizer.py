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

def normalize_text(text):
    """
    Normalización básica previa al tokenizado.
    """
    text = text.lower()
    text = translate(text)
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("’", "'").replace("`", "'")
    return text

def tokenize(text):
    """
    Tokeniza el texto en varias etapas:
    1. fechas
    2. números
    3. abreviaturas
    4. palabras comunes

    Devuelve una lista simple de tokens, sin tipo.
    """

    text = normalize_text(text)

    # Patrones
    date_re = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
    number_re = re.compile(r"\b\d+(?:[.,]\d+)*(?:%|\b)")
    abbrev_re = re.compile(r"\b(?:[a-z]\.){2,}|(?:[a-z]{1,10}\.)")
    word_re = re.compile(r"\b[a-z]+(?:[-'][a-z]+)*\b")

    tokens = []

    # Copia del texto sobre la que vamos "tapando" matches
    remaining = text

    def extract_and_mask(pattern, s, token_list):
        """
        Extrae matches de pattern, los agrega a token_list
        y los reemplaza por espacios en el string.
        """
        matches = list(pattern.finditer(s))
        for m in matches:
            token_list.append(m.group())

        # reemplazar de derecha a izquierda para no romper índices
        chars = list(s)
        for m in reversed(matches):
            start, end = m.span()
            chars[start:end] = " " * (end - start)

        return "".join(chars)

    # 1) fechas
    remaining = extract_and_mask(date_re, remaining, tokens)
    # 2) números
    remaining = extract_and_mask(number_re, remaining, tokens)
    # 3) abreviaturas
    remaining = extract_and_mask(abbrev_re, remaining, tokens)
    # 4) palabras comunes
    tokens.extend(word_re.findall(remaining))

    return tokens