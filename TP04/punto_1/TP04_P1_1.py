#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP04 - Recuperación de posting list desde el índice BSBI

Uso:
    python TP04_P1_1.py <output_dir> <término>

Parámetros:
    output_dir   Directorio de salida generado por TP04_P1.py
                 (debe contener vocabulary.pkl, doc2file.pkl e index.bin).
    término      Palabra a buscar en el índice.

Ejemplo:
    python TP04_P1_1.py output/ python
    python TP04_P1_1.py output/ "machine"

Salida por cada documento encontrado:
    DocName:docID:Frecuencia
"""

import os
import sys
import struct
import pickle

# Formato del índice con frecuencia: (docid, freq) — 2 unsigned ints big-endian
IDX_FMT_FREQ  = ">II"   # 8 bytes
IDX_FMT_NFREQ = ">I"    # 4 bytes
IDX_POST_FREQ  = struct.calcsize(IDX_FMT_FREQ)   # 8
IDX_POST_NFREQ = struct.calcsize(IDX_FMT_NFREQ)  # 4


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def retrieve_posting(index_path: str, seek: int, df: int, has_freq: bool) -> list:
    """
    Lee la posting list de un término desde index.bin.

    Retorna una lista de tuplas:
        - (docid, freq)  si has_freq=True
        - (docid, None)  si has_freq=False
    """
    postings = []
    fmt      = IDX_FMT_FREQ  if has_freq else IDX_FMT_NFREQ
    rec_size = IDX_POST_FREQ  if has_freq else IDX_POST_NFREQ

    with open(index_path, "rb") as f:
        f.seek(seek)
        for _ in range(df):
            raw = f.read(rec_size)
            if len(raw) < rec_size:
                break
            values = struct.unpack(fmt, raw)
            if has_freq:
                postings.append((values[0], values[1]))   # (docid, freq)
            else:
                postings.append((values[0], None))        # (docid, sin freq)

    return postings


def detect_has_freq(index_path: str, vocabulary: dict) -> bool:
    """
    Infiere si el índice almacena frecuencias comparando el tamaño del archivo
    con lo que esperaríamos para 4 u 8 bytes por posting.
    """
    total_df = sum(entry[1] for entry in vocabulary.values())
    if total_df == 0:
        return True  # no se puede determinar, asume freq

    idx_size = os.path.getsize(index_path)
    size_if_freq  = total_df * IDX_POST_FREQ
    size_if_nfreq = total_df * IDX_POST_NFREQ

    # Elegimos el que coincida exactamente
    if idx_size == size_if_freq:
        return True
    if idx_size == size_if_nfreq:
        return False

    # Por defecto, asume freq
    return True


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(0)

    output_dir = sys.argv[1]
    term       = sys.argv[2].lower()   # el índice almacena términos en minúsculas

    vocab_path    = os.path.join(output_dir, "vocabulary.pkl")
    doc2file_path = os.path.join(output_dir, "doc2file.pkl")
    index_path    = os.path.join(output_dir, "index.bin")

    # --- Validaciones 
    for path, name in [(vocab_path, "vocabulary.pkl"),
                       (doc2file_path, "doc2file.pkl"),
                       (index_path, "index.bin")]:
        if not os.path.isfile(path):
            print(f"[ERROR] No se encontró '{name}' en '{output_dir}'.")
            print("        Asegurate de haber ejecutado TP04_P1.py primero.")
            sys.exit(1)

    # --- Cargar estructuras 
    vocabulary: dict = load_pickle(vocab_path)   # {term: [seek, df, term_id]}
    doc2file:   dict = load_pickle(doc2file_path) # {docid: rel_path}

    # --- Buscar el término 
    if term not in vocabulary:
        print(f"El término '{term}' no se encuentra en el vocabulario.")
        sys.exit(0)

    seek, df, term_id = vocabulary[term]
    has_freq = detect_has_freq(index_path, vocabulary)

    # --- Recuperar la posting list 
    postings = retrieve_posting(index_path, seek, df, has_freq)

    # --- Mostrar resultados 
    print(f"Término: '{term}'  |  DF = {df}  |  term_id = {term_id}\n")
    print(f"{'DocName':<50} {'docID':>7}  {'Frecuencia':>10}")
    print("-" * 72)

    for docid, freq in postings:
        doc_name = doc2file.get(docid, "<desconocido>")
        freq_str = str(freq) if freq is not None else "N/A"
        print(f"{doc_name:<50} {docid:>7}  {freq_str:>10}")

    print("-" * 72)
    print(f"Total de documentos: {len(postings)}")


if __name__ == "__main__":
    main()
