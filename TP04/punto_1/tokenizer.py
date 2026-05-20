import re
import unicodedata

def translate(string):
    return (
        unicodedata.normalize("NFD", string)
        .encode("ascii", "ignore")
        .decode("ascii")
    )

# 1) URLs  (con algunas palabras fijas: http, https, ftp, sftp, www.)
_URL_RE = re.compile(
    r'(?:(?:https?|ftp|sftp)://|www\.)[^\s<>"\']+'  ,
    re.IGNORECASE
)

# 2) Emails
_EMAIL_RE = re.compile(
    r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}\b'
)

# 3) Fechas
_DATE_RE = re.compile(
    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
)

# 4) Números
# acepta enteros, decimales, miles, porcentajes, teléfonos simples
_NUMBER_RE = re.compile(
    r'\b(?:\+?\d{1,3}[\s-]?)?(?:\d+[.,:/-]?)+%?\b'
)

# 5) Abreviaturas/acrónimos — orden de mayor a menor especificidad:
# a) Acrónimos compuestos con puntos internos: U.S.A.  EE.UU.  D.A.S.M.I
#    (dos o más segmentos  letra(s) + punto, con punto final opcional)
# b) Abreviaturas simples: mayúscula + hasta 5 letras (cualquiera) + punto
#    Captura: Dr.  Sra.  Srta.  Dra.  Lic.  Prof.  Ing.  etc.
#    Trade-off: palabras cortas al fin de oración (Casa., Esta.) también matchean.
# c) Letra sola con punto: A.  (la menos específica, va última)
_ABBREV_RE = re.compile(
    r'(?:[A-Za-záéíóúüñÁÉÍÓÚÜÑ]{1,4}\.){2,}[A-Za-záéíóúüñÁÉÍÓÚÜÑ]{0,4}'  # a) EE.UU.  U.S.A
    r'|\b[A-ZÁÉÍÓÚÜÑ][A-Za-záéíóúüñ]{1,5}\.'                                # b) Dr.  Sra.  Lic.
    r'|\b[A-Za-z]\.(?=[^\s])'                                                # c) p. (pieza), i.
)

# 6) Nombres propios genéricos:
# permite conectores en minúscula entre palabras capitalizadas
_CONNECTORS = r'(?:de|del|la|las|los|y|e)'
_PROPER_RE = re.compile(                                 # no tras fin de oración
    rf'\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+'
    rf'(?:[ \t]{{1,3}}(?:{_CONNECTORS})[ \t]{{1,3}}[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+'
    rf'|[ \t]{{1,3}}[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)+'
)

# 7) Palabras comunes
_WORD_RE = re.compile(
    r"\b[a-z]+(?:[-'][a-z]+)*\b"
)

def _extract_and_mask(pattern, text, token_type):
    """
    Encuentra matches, los devuelve como [(pos, token, tipo)] y los enmascara.
    """
    matches = list(pattern.finditer(text))
    found = [(m.start(), m.group(), token_type) for m in matches]

    chars = list(text)
    for m in reversed(matches):
        chars[m.start():m.end()] = ' ' * (m.end() - m.start())

    return "".join(chars), found

def tokenize(text):
    """
    Tokenizer basado en los criterios de Grefenstette & Tapanainen.

    Retorna una lista de tuplas (token, tipo) donde tipo es uno de:
        'url'    - URL completa
        'email'  - dirección de correo
        'date'   - fecha (dd/mm/aaaa)
        'number' - número, teléfono, porcentaje
        'abbrev' - abreviatura o acrónimo (Dr., S.A., EE.UU., …)
        'proper' - nombre propio multi-palabra (villa carlos paz)
        'word'   - palabra común
    """
    tokens = []
    remaining = text

    # 1. URLs (se conservan en su forma original)
    remaining, found = _extract_and_mask(_URL_RE, remaining, "url")
    tokens.extend((tok, t) for _, tok, t in found)

    # 2. Emails (se conservan en su forma original)
    remaining, found = _extract_and_mask(_EMAIL_RE, remaining, "email")
    tokens.extend((tok, t) for _, tok, t in found)

    # 3. Fechas (se conservan tal cual)
    remaining, found = _extract_and_mask(_DATE_RE, remaining, "date")
    tokens.extend((tok, t) for _, tok, t in found)

    # 4. Números (se conservan tal cual)
    remaining, found = _extract_and_mask(_NUMBER_RE, remaining, "number")
    tokens.extend((tok, t) for _, tok, t in found)

    # 5. Abreviaturas (ordenadas por posición; se normalizan a minúsculas)
    remaining, found = _extract_and_mask(_ABBREV_RE, remaining, "abbrev")
    tokens_with_pos = [(pos, translate(tok), t) for pos, tok, t in found]

    # 6. Nombres propios (detección ANTES de pasar a minúsculas)
    remaining, found = _extract_and_mask(_PROPER_RE, remaining, "proper")
    tokens_with_pos += [
        (pos, translate(tok).strip(), t)
        for pos, tok, t in found
    ]

    # Ordenar abreviaturas + propios por posición y agregar sin posición
    tokens_with_pos.sort(key=lambda x: x[0])
    tokens.extend((tok, t) for _, tok, t in tokens_with_pos)

    # 7. Palabras comunes (sobre el texto restante, ya normalizado)
    remaining_norm = translate(remaining.lower())
    tokens.extend((m.group(), "word") for m in _WORD_RE.finditer(remaining_norm))

    return tokens