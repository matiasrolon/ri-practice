#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP04 - Punto 2: Búsqueda booleana TAAT sobre índice BSBI

Uso:
    python TP04_P2.py <index_dir> "<consulta>"
    python TP04_P2.py <index_dir> "<consulta>" --freq

Parámetros:
    index_dir       Directorio de salida del index (TP04_P1.py), debe contener
                    index.bin, vocabulary.pkl, doc2file.pkl.
    consulta        Expresión booleana entre comillas. Operadores: AND, OR, NOT.
                    Los términos van en minúscula, los operadores en MAYÚSCULA.
                    Se permiten paréntesis para agrupar.
    --freq          Indicar si el índice fue creado con frecuencias (8 bytes/posting).
                    Si no se indica, se asume solo docID (4 bytes/posting).

Ejemplos:
    python TP04_P2.py ../punto_1/output "((python AND code) OR linux)"
    python TP04_P2.py ../punto_1/output "((house AND NOT cat) OR NOT dog)" --freq
    python TP04_P2.py ../punto_1/output "dinosaur AND NOT animal"

Estrategia: TAAT (Term-At-A-Time)
"""

import os
import sys
import struct
import pickle
from os.path import join, isdir, isfile

import boolean  # pip install boolean.py


# ── Constantes de formato (deben coincidir con TP04_P1.py) ────────────────────
IDX_FMT_FREQ  = ">II"   # 8 bytes: (docid, freq)
IDX_FMT_NFREQ = ">I"    # 4 bytes: (docid,)
IDX_POST_FREQ  = struct.calcsize(IDX_FMT_FREQ)   # 8
IDX_POST_NFREQ = struct.calcsize(IDX_FMT_NFREQ)  # 4


# ══════════════════════════════════════════════════════════════════════════════
# Carga del índice
# ══════════════════════════════════════════════════════════════════════════════

def load_index(index_dir: str):
    """Carga vocabulary.pkl y doc2file.pkl desde el directorio del índice."""
    vocab_path    = join(index_dir, "vocabulary.pkl")
    doc2file_path = join(index_dir, "doc2file.pkl")

    for p, name in [(vocab_path, "vocabulary.pkl"), (doc2file_path, "doc2file.pkl")]:
        if not isfile(p):
            print(f"[ERROR] No se encontró '{name}' en '{index_dir}'.")
            sys.exit(1)

    with open(vocab_path, "rb") as f:
        vocabulary = pickle.load(f)  # {term_str: [seek, df, term_id]}

    with open(doc2file_path, "rb") as f:
        doc2file = pickle.load(f)    # {docid_int: rel_path_str}

    return vocabulary, doc2file


def load_posting_list(index_path: str, seek: int, df: int, store_freq: bool) -> set:
    """
    Lee la posting list de un término desde index.bin.
    Retorna un set de docIDs.
    """
    post_size = IDX_POST_FREQ if store_freq else IDX_POST_NFREQ
    fmt       = IDX_FMT_FREQ if store_freq else IDX_FMT_NFREQ
    n_ints    = 2 if store_freq else 1

    docids = set()
    with open(index_path, "rb") as f:
        f.seek(seek)
        raw = f.read(df * post_size)

    offset = 0
    for _ in range(df):
        values = struct.unpack_from(fmt, raw, offset)
        docids.add(values[0])  # docid siempre es el primer valor
        offset += post_size

    return docids


# ══════════════════════════════════════════════════════════════════════════════
# Evaluación usando boolean.py
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_query(
    query_str: str,
    vocabulary: dict,
    index_path: str,
    all_docids: set,
    store_freq: bool,
) -> set:
    """
    Parsea la consulta booleana con boolean.py y la evalúa usando TAAT.
    """
    algebra = boolean.BooleanAlgebra()

    # Parsear la expresión
    try:
        expr = algebra.parse(query_str)
    except boolean.ParseError as e:
        print(f"[ERROR] Expresión booleana inválida: {e}")
        sys.exit(1)

    # Cache de posting lists ya cargadas
    cache: dict[str, set] = {}

    def resolve(node) -> set:
        """Resuelve recursivamente el árbol booleano."""

        # Hoja: símbolo en boolean.py (es un término)
        if isinstance(node, boolean.Symbol):
            term = str(node.obj).lower().strip()
            if term in cache:
                return cache[term]
            if term not in vocabulary:
                print(f"[WARN] Término '{term}' no existe en el vocabulario. Se trata como conjunto vacío.")
                cache[term] = set()
                return set()
            seek, df, _ = vocabulary[term]
            docids = load_posting_list(index_path, seek, df, store_freq)
            cache[term] = docids
            return docids

        # NOT: complemento
        if isinstance(node, boolean.NOT):
            child_set = resolve(node.args[0])
            return all_docids - child_set

        # AND: intersección
        if isinstance(node, boolean.AND):
            result = None
            for child in node.args:
                child_set = resolve(child)
                result = child_set if result is None else result & child_set
            return result

        # OR: unión
        if isinstance(node, boolean.OR):
            result = set()
            for child in node.args:
                result = result | resolve(child)
            return result

        # TRUE / FALSE literales del álgebra
        if node == algebra.TRUE:
            return all_docids
        if node == algebra.FALSE:
            return set()

        print(f"[ERROR] Nodo no reconocido: {type(node)} → {node}")
        return set()

    return resolve(expr)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(0)

    index_dir = sys.argv[1]
    query_str = sys.argv[2]
    store_freq = "--freq" in sys.argv

    if not isdir(index_dir):
        print(f"[ERROR] '{index_dir}' no es un directorio válido.")
        sys.exit(1)

    index_path = join(index_dir, "index.bin")
    if not isfile(index_path):
        print(f"[ERROR] No se encontró 'index.bin' en '{index_dir}'.")
        sys.exit(1)

    # Cargar índice
    vocabulary, doc2file = load_index(index_dir)
    all_docids = set(doc2file.keys())

    print(f"[INFO] Índice cargado: {len(vocabulary)} términos, {len(all_docids)} documentos.")
    print(f"[INFO] Modo: {'docID+freq' if store_freq else 'solo docID'}")
    print(f"[INFO] Consulta: {query_str}\n")

    # Evaluar
    result_docids = evaluate_query(query_str, vocabulary, index_path, all_docids, store_freq)

    # Mostrar resultados
    if not result_docids:
        print("Sin resultados.")
        return

    sorted_results = sorted(result_docids)
    print(f"Documentos encontrados: {len(sorted_results)}\n")
    print(f"{'docID':>8}  {'Archivo'}")
    print(f"{'─' * 8}  {'─' * 50}")
    for docid in sorted_results:
        filename = doc2file.get(docid, "???")
        print(f"{docid:>8}  {filename}")


if __name__ == "__main__":
    main()
