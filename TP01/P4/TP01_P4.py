#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP01 - Punto 4

Uso:
    python TP01_P4.py <dir_coleccion> [--stopwords <archivo_stopwords>]

Parámetros:
    dir_coleccion           Directorio con los documentos .txt de la colección.
    --stopwords <archivo>   (Opcional) Archivo con palabras vacías a eliminar.

Ejemplo:
(parado en la carpeta del punto, asumiendo que se dejan las colecciones dentro de la carpeta 'data' en la raiz)
    python TP01_P4.py ../data/RI-tknz-data/
    python TP01_P4.py ../data/RI-tknz-data/ --stopwords ../data/stopwords/spanish.txt

Archivos generados:
    terminos.txt    - Lista ordenada de términos con CF y DF.
    estadisticas.txt - Estadísticas generales de la colección.
    frecuencias.txt  - Top 10 más y menos frecuentes.
"""

import os
import sys
from os import listdir
from os.path import join, isdir, isfile
from typing import List, Tuple
from tokenizer import tokenize  # type: ignore[import]

# ── Configuración ─────────────────────────────────────────────────────────────
MIN_TERM_LEN = 2   # longitud mínima de un término (inclusive)
MAX_TERM_LEN = 25  # longitud máxima de un término (inclusive)
# ─────────────────────────────────────────────────────────────────────────────


def load_stopwords(filepath: str) -> set:
    """Carga las palabras vacías desde un archivo (una por línea)."""
    stopwords = set()
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                stopwords.add(word)
    return stopwords


def is_valid_term(term: str, stopwords: set) -> bool:
    """Filtra un término por longitud y por pertenencia al listado de stopwords."""
    if len(term) < MIN_TERM_LEN or len(term) > MAX_TERM_LEN:
        return False
    if term in stopwords:
        return False
    return True


def parse_args():
    """Parsea los argumentos de línea de comandos."""
    dir_path = "../RI-tknz-data/"  # default path
    stopwords_file = None

    if len(sys.argv) < 2:
        print("[INFO] No se pasaron parámetros. Se usará el directorio por defecto.")
    else:
        dir_path = sys.argv[1]
        if "--stopwords" in sys.argv:
            idx = sys.argv.index("--stopwords")
            if idx + 1 < len(sys.argv):
                stopwords_file = sys.argv[idx + 1]
            else:
                print("[ERROR] Se indicó --stopwords pero no se proporcionó el archivo.")
                sys.exit(1)

    return dir_path, stopwords_file


def process_collection(dir_path: str, stopwords: set):
    """
    Procesa todos los documentos .txt del directorio.

    Retorna:
        cf          dict  {término: collection_frequency}
        df          dict  {término: document_frequency}
        doc_stats   list  [(tokens_count, terms_count), ...] por documento
    """
    cf = {}      # collection frequency
    df = {}      # document frequency
    doc_stats = []  # (token_count, term_count) por documento

    files = sorted(f for f in listdir(dir_path) if f.endswith(".txt"))

    for filename in files:
        path = join(dir_path, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()

        # Tokenización
        raw_tokens = tokenize(content)
        # Filtrado: aplicar restricciones de longitud y stopwords
        valid_tokens = [t for t in raw_tokens if is_valid_term(t, stopwords)]
        # Tokens del documento: todos los tokens crudos (antes del filtro de términos)
        token_count = len(raw_tokens)
        # Términos del documento: tokens válidos tras filtrado
        term_count = len(valid_tokens)
        # CF: frecuencia acumulada en la colección
        for term in valid_tokens:
            cf[term] = cf.get(term, 0) + 1
        # DF: cada término (unico) suma 1 al documento
        terms_in_doc = set(valid_tokens)
        for term in terms_in_doc:
            df[term] = df.get(term, 0) + 1

        doc_stats.append((token_count, term_count))

    return cf, df, doc_stats


def write_terms(cf: dict, df: dict, output_dir: str):
    """Genera terminos.txt con formato: <termino> <CF> <DF> ordenado alfabéticamente."""
    path = join(output_dir, "terminos.txt")
    sorted_terms = sorted(cf.keys())
    with open(path, "w", encoding="utf-8") as f:
        for term in sorted_terms:
            f.write(f"{term} {cf[term]} {df[term]}\n")
    print(f"[OK] terminos.txt generado ({len(sorted_terms)} términos).")
    return path


def write_statistics(cf: dict, df: dict, doc_stats: list, output_dir: str):
    """Genera estadisticas.txt con las métricas solicitadas."""
    path = join(output_dir, "estadisticas.txt")
    # cantidad de documentos procesados
    num_docs = len(doc_stats)
    # cantidad total de tokens en la colección
    total_tokens = sum(s[0] for s in doc_stats)
    # cantidad total de terminos en la colección
    total_terms_tokens = sum(s[1] for s in doc_stats)   # suma de tokens válidos
    # cantidad de terminos UNICOS en la colección
    num_unique_terms = len(cf)
    # promedio de tokens por documento
    avg_tokens = total_tokens / num_docs if num_docs else 0
    # promedio de terminos por documento
    avg_terms = total_terms_tokens / num_docs if num_docs else 0
    # Largo promedio de un término (promedio de la longitud de cada término único)
    avg_term_len = (
        sum(len(t) for t in cf) / num_unique_terms if num_unique_terms else 0
    )
    # Documento más corto y más largo por tokens (crudos)
    min_stats = min(doc_stats, key=lambda s: s[0])
    max_stats = max(doc_stats, key=lambda s: s[0])
    # Términos que aparecen sólo 1 vez (hapax legomena)
    hapax = sum(1 for v in cf.values() if v == 1)
    # Escribir estadísticas en el archivo
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{num_docs}\n")
        f.write(f"{total_tokens} {num_unique_terms}\n")
        f.write(f"{avg_tokens:.2f} {avg_terms:.2f}\n")
        f.write(f"{avg_term_len:.2f}\n") 
        f.write(f"{min_stats[0]} {min_stats[1]} {max_stats[0]} {max_stats[1]}\n")
        f.write(f"{hapax}\n")

    print(f"[OK] estadisticas.txt generado.")
    return path


def write_frequencies(cf: "dict[str, int]", output_dir: str):
    """Genera frecuencias.txt con los 10 más y 10 menos frecuentes."""
    path = join(output_dir, "frecuencias.txt")

    all_terms = sorted(cf.items(), key=lambda x: x[1], reverse=True)
    n = len(all_terms)

    top10 = list(all_terms[i] for i in range(min(10, n)))
    bottom10 = list(reversed([all_terms[i] for i in range(max(0, n - 10), n)]))

    with open(path, "w", encoding="utf-8") as f:
        f.write("=== 10 términos más frecuentes ===\n")
        for term, freq in top10:
            f.write(f"{term} {freq}\n")
        f.write("\n=== 10 términos menos frecuentes ===\n")
        for term, freq in bottom10:
            f.write(f"{term} {freq}\n")

    print(f"[OK] frecuencias.txt generado.")
    return path


def main():
    dir_path, stopwords_file = parse_args()

    # Validar directorio
    if not isdir(dir_path):
        print(f"[ERROR] '{dir_path}' no es un directorio válido.")
        sys.exit(1)

    # Cargar stopwords
    stopwords = set()
    if stopwords_file:
        if not isfile(stopwords_file):
            print(f"[ERROR] El archivo de stopwords '{stopwords_file}' no existe.")
            sys.exit(1)
        stopwords = load_stopwords(stopwords_file)
        print(f"[INFO] Stopwords cargadas: {len(stopwords)} palabras desde '{stopwords_file}'.")
    else:
        print("[INFO] No se eliminan palabras vacías.")

    print(f"[INFO] MIN_TERM_LEN={MIN_TERM_LEN}, MAX_TERM_LEN={MAX_TERM_LEN}")
    print(f"[INFO] Procesando colección en: {dir_path}\n")

    # Procesar colección
    cf, df, doc_stats = process_collection(dir_path, stopwords)

    if not doc_stats:
        print("[WARN] No se encontraron archivos .txt en el directorio.")
        sys.exit(0)

    # Directorio de salida: mismo que el script
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Generar archivos de salida
    write_terms(cf, df, output_dir)
    write_statistics(cf, df, doc_stats, output_dir)
    write_frequencies(cf, output_dir)
    print("\n[INFO] Archivos generados en: ", output_dir)

if __name__ == "__main__":
    main()