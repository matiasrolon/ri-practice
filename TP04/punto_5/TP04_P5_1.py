#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP04 - Punto 5.1: Recuperar skip list de un término

Uso:
    python TP04_P5_1.py <index_dir> <termino>

Parámetros:
    index_dir           Directorio del índice (vocabulary.pkl, index.bin, doc2file.pkl).
    term                Término a buscar

Ejemplo:
    python TP04_P5_1.py ../punto_1/output hotel

Salida:
    docName:docID  (ordenada por docName)
"""

import sys
import math
import struct
import pickle
from os.path import join, isdir, isfile

SKIP_FMT  = ">II"
SKIP_SIZE = struct.calcsize(SKIP_FMT)  # 8


def n_skips_for(df):
    if df <= 1:
        return 0
    step = max(1, int(math.sqrt(df)))
    return math.ceil(df / step)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(0)

    index_dir = sys.argv[1]
    term      = sys.argv[2].strip().lower()

    for name in ["vocabulary.pkl", "doc2file.pkl"]:
        if not isfile(join(index_dir, name)):
            print(f"[ERROR] No se encontró '{name}' en '{index_dir}'.")
            sys.exit(1)

    skips_bin_path = "./output/skips.bin"
    if not isfile(skips_bin_path):
        print(f"[ERROR] No se encontró 'skips.bin' en './output'.")
        sys.exit(1)

    with open(join(index_dir, "vocabulary.pkl"), "rb") as f:
        vocabulary = pickle.load(f)
    with open(join(index_dir, "doc2file.pkl"), "rb") as f:
        doc2file = pickle.load(f)

    if term not in vocabulary:
        print(f"[ERROR] Término '{term}' no existe en el vocabulario.")
        sys.exit(1)

    entry = vocabulary[term]
    if len(entry) < 4:
        print("[ERROR] El vocabulario no tiene skip_seek. Ejecute primero TP04_P5.py.")
        sys.exit(1)

    _, df, _, skip_seek = entry
    n = n_skips_for(df)

    if n == 0:
        print(f"Término '{term}' (df={df}): sin skip pointers (lista demasiado corta).")
        return

    with open(skips_bin_path, "rb") as f:
        f.seek(skip_seek)
        raw = f.read(n * SKIP_SIZE)

    results = []
    for i in range(n):
        docid, _ = struct.unpack_from(SKIP_FMT, raw, i * SKIP_SIZE)
        doc_name = doc2file.get(docid, "???")
        results.append((doc_name, docid))

    results.sort(key=lambda x: x[0])

    print(f"Término '{term}' (df={df}, {n} skip pointers):\n")
    for doc_name, docid in results:
        print(f"{doc_name}:{docid}")


if __name__ == "__main__":
    main()
