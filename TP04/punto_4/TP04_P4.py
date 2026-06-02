#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP04 - Punto 4: Ranking DAAT con modelo vectorial (coseno)

Uso:
    python TP04_P4.py <index_dir> "<query>" [opciones]

Parámetros:
    index_dir       Directorio del índice (index.bin, vocabulary.pkl, doc2file.pkl).
                    El índice DEBE haber sido creado con --freq (8 bytes/posting).
    query           Términos de la query separados por espacio.
    --k <int>       Cantidad de documentos a retornar (default: 10).

Ejemplo:
    python TP04_P4.py ./index "python algorithm data"
    python TP04_P4.py ./index "car motor america" --k 5

Salida:
    DocName:docID:Score   (ordenado por score descendente)

Pesos:
    w(t,d) = (1 + log10(tf_td))  ×  log10(N / df_t)
    w(t,q) = log10(N / df_t)     (asumiendo tf_tq = 1 para cada término)
    score  = Σ w(t,q) × w(t,d)  /  (|d| × |q|)

    |d| se calcula escaneando las posting lists de los términos de la query.
    |q| = sqrt(Σ w(t,q)²).
"""

import sys
import struct
import pickle
import math
import heapq
from os.path import join, isdir, isfile


# ── Constantes de formato ─────────────────────────────────────────────────────
IDX_FMT  = ">II"                             # (docid, freq)
IDX_POST = struct.calcsize(IDX_FMT)          # 8 bytes


# ══════════════════════════════════════════════════════════════════════════════
# Carga del índice
# ══════════════════════════════════════════════════════════════════════════════

def load_index(index_dir: str):
    vocab_path    = join(index_dir, "vocabulary.pkl")
    doc2file_path = join(index_dir, "doc2file.pkl")

    for p, name in [(vocab_path, "vocabulary.pkl"), (doc2file_path, "doc2file.pkl")]:
        if not isfile(p):
            print(f"[ERROR] No se encontró '{name}' en '{index_dir}'.")
            sys.exit(1)

    with open(vocab_path, "rb") as f:
        vocabulary = pickle.load(f)

    with open(doc2file_path, "rb") as f:
        doc2file = pickle.load(f)

    return vocabulary, doc2file


def load_posting_list(index_path: str, seek: int, df: int) -> list:
    """
    Lee una posting list ordenada por docID desde index.bin.
    Retorna [(docid, freq), ...] — ya ordenada por docID.
    """
    with open(index_path, "rb") as f:
        f.seek(seek)
        raw = f.read(df * IDX_POST)

    postings = []
    offset = 0
    for _ in range(df):
        docid, freq = struct.unpack_from(IDX_FMT, raw, offset)
        postings.append((docid, freq))
        offset += IDX_POST
    return postings


# ══════════════════════════════════════════════════════════════════════════════
# DAAT con coseno
# ══════════════════════════════════════════════════════════════════════════════

def daat_cosine(query_terms: list, vocabulary: dict, index_path: str, n_docs: int, k: int) -> list:
    """
    Ejecuta DAAT sobre las posting lists de los términos de la query.

    Retorna [(score, docid), ...] ordenado por score descendente, hasta k elementos.
    """

    # ── 1. Filtrar términos que existen en el vocabulario ─────────────────────
    valid_terms = []
    for t in query_terms:
        if t in vocabulary:
            valid_terms.append(t)
        else:
            print(f"[WARN] Término '{t}' no existe en el vocabulario, se ignora.")

    if not valid_terms:
        return []

    # ── 2. Cargar posting lists y calcular IDF ────────────────────────────────
    lists = []      # [(idf, [(docid, freq), ...]), ...]
    query_weights = []

    for t in valid_terms:
        seek, df, _ = vocabulary[t]
        postings = load_posting_list(index_path, seek, df)
        idf = math.log10(n_docs / df) if df > 0 else 0.0
        lists.append((idf, postings))
        query_weights.append(idf)  # w(t,q) = idf (asumiendo tf_tq = 1)

    # Norma del vector query
    query_norm = math.sqrt(sum(w * w for w in query_weights))
    if query_norm == 0:
        return []

    # ── 3. Norma de documentos ──
    # |d| = sqrt(Σ_t w(t,d)²) donde w(t,d) = (1 + log10(tf)) * idf
    doc_norm_sq: dict[int, float] = {}
    for idf, postings in lists:
        for docid, freq in postings:
            w_td = (1.0 + math.log10(freq)) * idf if freq > 0 else 0.0
            doc_norm_sq[docid] = doc_norm_sq.get(docid, 0.0) + w_td * w_td

    # ── 4. DAAT: recorrido paralelo por docID ─────────────────────────────────
    # Cursores: un índice por lista
    cursors = [0] * len(lists)
    # Min-heap de tamaño k: elementos = (score, docid)  (min-heap, el peor arriba)
    top_k: list[tuple[float, int]] = []

    while True:
        # Encontrar el menor docID actual entre todos los cursors activos
        min_docid = None
        for i, (_, postings) in enumerate(lists):
            if cursors[i] < len(postings):
                did = postings[cursors[i]][0]
                if min_docid is None or did < min_docid:
                    min_docid = did

        if min_docid is None:
            break  # todas las listas agotadas

        # Acumular score para min_docid desde todas las listas que lo contienen
        score = 0.0
        for i, (idf, postings) in enumerate(lists):
            if cursors[i] < len(postings) and postings[cursors[i]][0] == min_docid:
                freq = postings[cursors[i]][1]
                w_td = (1.0 + math.log10(freq)) * idf if freq > 0 else 0.0
                w_tq = query_weights[i]
                score += w_tq * w_td
                cursors[i] += 1  # avanzar cursor

        # Normalizar por normas de doc y query
        d_norm = math.sqrt(doc_norm_sq.get(min_docid, 1.0))
        if d_norm > 0 and query_norm > 0:
            score /= (d_norm * query_norm)

        # Mantener top-k con min-heap
        if len(top_k) < k:
            heapq.heappush(top_k, (score, min_docid))
        elif score > top_k[0][0]:
            heapq.heapreplace(top_k, (score, min_docid))

    # Ordenar por score descendente
    top_k.sort(key=lambda x: x[0], reverse=True)
    return top_k


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(0)

    index_dir = sys.argv[1]
    query_str = sys.argv[2]
    k = 10

    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--k" and i + 1 < len(sys.argv):
            k = int(sys.argv[i + 1]); i += 2
        else:
            i += 1

    if not isdir(index_dir):
        print(f"[ERROR] '{index_dir}' no es un directorio válido.")
        sys.exit(1)

    index_path = join(index_dir, "index.bin")
    if not isfile(index_path):
        print(f"[ERROR] 'index.bin' no encontrado en '{index_dir}'.")
        sys.exit(1)

    vocabulary, doc2file = load_index(index_dir)
    n_docs = len(doc2file)

    query_terms = query_str.strip().lower().split()
    print(f"[INFO] Índice: {len(vocabulary)} términos, {n_docs} documentos.")
    print(f"[INFO] Query: {query_terms}")
    print(f"[INFO] Top-k: {k}\n")

    results = daat_cosine(query_terms, vocabulary, index_path, n_docs, k)

    if not results:
        print("Sin resultados.")
        return

    for score, docid in results:
        doc_name = doc2file.get(docid, "???")
        print(f"{doc_name}:{docid}:{score:.6f}")


if __name__ == "__main__":
    main()
