#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP04 - Punto 3: Corridas de consultas booleanas TAAT

Uso:
    python TP04_P3.py <index_dir> <queries_file> [opciones]

Parámetros:
    index_dir           Directorio del índice (vocabulary.pkl, index.bin, doc2file.pkl).
    queries_file        Archivo de queries (formato "id:texto", una por línea).
    --freq              El índice fue creado con frecuencias (8 bytes/posting).
    --in-memory         Cargar index.bin completo en RAM antes de ejecutar.

Ejemplo:
    python TP04_P3.py ./index EFF-10K-queries.txt
    python TP04_P3.py ./index EFF-10K-queries.txt --freq --in-memory

Archivo generado (en index_dir):
    query_results.csv   Cada línea: <patron>,<total_postings>,<tiempo_seg>
"""

import os
import sys
import struct
import pickle
import time
from os.path import join, isdir, isfile

import boolean


# ── Constantes de formato (deben coincidir con TP04_P1.py) ────────────────────
IDX_FMT_FREQ   = ">II"
IDX_FMT_NFREQ  = ">I"
IDX_POST_FREQ  = struct.calcsize(IDX_FMT_FREQ)   # 8
IDX_POST_NFREQ = struct.calcsize(IDX_FMT_NFREQ)  # 4


# ══════════════════════════════════════════════════════════════════════════════
# Carga del índice
# ══════════════════════════════════════════════════════════════════════════════

def load_index_metadata(index_dir: str):
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


def load_posting_list_disk(index_path: str, seek: int, df: int, store_freq: bool) -> set:
    post_size = IDX_POST_FREQ if store_freq else IDX_POST_NFREQ
    fmt       = IDX_FMT_FREQ if store_freq else IDX_FMT_NFREQ
    with open(index_path, "rb") as f:
        f.seek(seek)
        raw = f.read(df * post_size)
    docids = set()
    offset = 0
    for _ in range(df):
        docids.add(struct.unpack_from(fmt, raw, offset)[0])
        offset += post_size
    return docids


def load_posting_list_mem(index_bytes: bytes, seek: int, df: int, store_freq: bool) -> set:
    post_size = IDX_POST_FREQ if store_freq else IDX_POST_NFREQ
    fmt       = IDX_FMT_FREQ if store_freq else IDX_FMT_NFREQ
    docids = set()
    offset = seek
    for _ in range(df):
        docids.add(struct.unpack_from(fmt, index_bytes, offset)[0])
        offset += post_size
    return docids


# ══════════════════════════════════════════════════════════════════════════════
# Evaluación booleana
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_query(query_str, vocabulary, all_docids, store_freq, load_fn, algebra):
    try:
        expr = algebra.parse(query_str)
    except boolean.ParseError:
        return set()

    cache = {}

    def resolve(node):
        if isinstance(node, boolean.Symbol):
            term = str(node.obj).lower().strip()
            if term in cache:
                return cache[term]
            if term not in vocabulary:
                cache[term] = set()
                return set()
            seek, df, _ = vocabulary[term]
            docids = load_fn(seek, df)
            cache[term] = docids
            return docids
        if isinstance(node, boolean.NOT):
            return all_docids - resolve(node.args[0])
        if isinstance(node, boolean.AND):
            result = None
            for child in node.args:
                s = resolve(child)
                result = s if result is None else result & s
            return result
        if isinstance(node, boolean.OR):
            result = set()
            for child in node.args:
                result |= resolve(child)
            return result
        if node == algebra.TRUE:
            return all_docids
        if node == algebra.FALSE:
            return set()
        return set()

    return resolve(expr)


# ══════════════════════════════════════════════════════════════════════════════
# Parsing de queries
# ══════════════════════════════════════════════════════════════════════════════

def parse_queries_file(filepath):
    queries = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                qid, text = line.split(":", 1)
                qid = qid.strip()
            else:
                qid = str(len(queries) + 1)
                text = line
            terms = text.strip().lower().split()
            if terms:
                queries.append((qid, terms))
    return queries


def filter_queries(queries, vocabulary, n_terms):
    return [
        (qid, terms) for qid, terms in queries
        if len(terms) == n_terms and all(t in vocabulary for t in terms)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Patrones booleanos
# ══════════════════════════════════════════════════════════════════════════════

PATTERNS_2 = {
    "t1 AND t2":       lambda t: f"({t[0]} AND {t[1]})",
    "t1 OR t2":        lambda t: f"({t[0]} OR {t[1]})",
    "t1 AND NOT t2":   lambda t: f"({t[0]} AND NOT {t[1]})",
}

PATTERNS_3 = {
    "t1 AND t2 AND t3":      lambda t: f"(({t[0]} AND {t[1]}) AND {t[2]})",
    "(t1 OR t2) AND NOT t3": lambda t: f"(({t[0]} OR {t[1]}) AND NOT {t[2]})",
    "(t1 AND t2) OR t3":     lambda t: f"(({t[0]} AND {t[1]}) OR {t[2]})",
}


# ══════════════════════════════════════════════════════════════════════════════
# Ejecución de queries
# ══════════════════════════════════════════════════════════════════════════════

def total_posting_size(terms, vocabulary):
    return sum(vocabulary[t][1] for t in terms)


def run_queries(queries, patterns, vocabulary, all_docids, store_freq, load_fn, algebra):
    """
    Retorna {patron: [(q_terms, total_postings, time_sec, n_results), ...]}.
    """
    results = {p: [] for p in patterns}
    for _, terms in queries:
        tp = total_posting_size(terms, vocabulary)
        for pat_name, pat_fn in patterns.items():
            query_str = pat_fn(terms)
            t0 = time.perf_counter()
            res = evaluate_query(query_str, vocabulary, all_docids, store_freq, load_fn, algebra)
            elapsed = time.perf_counter() - t0
            results[pat_name].append((terms, tp, elapsed, len(res)))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Salida
# ══════════════════════════════════════════════════════════════════════════════

def print_results(results_2, results_3, mode_label):
    """Imprime estadísticas por patrón a stdout."""
    for label, results in [("Queries |q|=2", results_2), ("Queries |q|=3", results_3)]:
        print(f"\n── {label} ─────────────────────────────────────────────")
        for pat_name, data in results.items():
            if not data:
                print(f"  {pat_name}: sin queries válidas")
                continue
            times    = [d[2] for d in data]
            postings = [d[1] for d in data]
            print(f"\n  Patrón: {pat_name}")
            print(f"    Queries ejecutadas:  {len(data)}")
            print(f"    Tiempo total:        {sum(times):.6f} s")
            print(f"    Tiempo promedio:     {sum(times)/len(times):.6f} s")
            print(f"    Tiempo mínimo:       {min(times):.6f} s")
            print(f"    Tiempo máximo:       {max(times):.6f} s")
            print(f"    Posting total prom:  {sum(postings)/len(postings):.1f}")


def write_results_file(results_2, results_3, output_path):
    """
    Escribe query_results.csv.
    Cada línea: query,patron,total_postings,tiempo_seg
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("query,patron,total_postings,tiempo_seg\n")
        for results in [results_2, results_3]:
            for pat_name, data in results.items():
                for q_terms, tp, elapsed, _ in data:
                    query_str = " ".join(q_terms)
                    f.write(f"{query_str},{pat_name},{tp},{elapsed:.9f}\n")
    print(f"[OK] {os.path.basename(output_path)} generado ({output_path}).")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(0)

    index_dir    = sys.argv[1]
    queries_file = sys.argv[2]
    store_freq   = "--freq" in sys.argv
    in_memory    = "--in-memory" in sys.argv

    if not isdir(index_dir):
        print(f"[ERROR] '{index_dir}' no es un directorio válido.")
        sys.exit(1)
    if not isfile(queries_file):
        print(f"[ERROR] '{queries_file}' no encontrado.")
        sys.exit(1)

    index_path = join(index_dir, "index.bin")
    if not isfile(index_path):
        print(f"[ERROR] 'index.bin' no encontrado en '{index_dir}'.")
        sys.exit(1)

    # ── Cargar índice ─────────────────────────────────────────────────────────
    vocabulary, doc2file = load_index_metadata(index_dir)
    all_docids = set(doc2file.keys())
    mode_label = "en memoria" if in_memory else "en disco"

    if in_memory:
        with open(index_path, "rb") as f:
            index_bytes = f.read()
        load_fn = lambda seek, df: load_posting_list_mem(index_bytes, seek, df, store_freq)
        print(f"[INFO] index.bin cargado en memoria ({len(index_bytes)/1e6:.2f} MB).")
    else:
        load_fn = lambda seek, df: load_posting_list_disk(index_path, seek, df, store_freq)
        print(f"[INFO] Modo disco (index.bin se lee por query).")

    print(f"[INFO] Vocabulario: {len(vocabulary)} términos, {len(all_docids)} documentos.")
    print(f"[INFO] Modo postings: {'docID+freq' if store_freq else 'solo docID'}")

    # ── Cargar y filtrar queries ──────────────────────────────────────────────
    all_queries = parse_queries_file(queries_file)
    print(f"[INFO] Queries leídas: {len(all_queries)}")

    q2 = filter_queries(all_queries, vocabulary, 2)
    q3 = filter_queries(all_queries, vocabulary, 3)
    print(f"[INFO] Queries |q|=2 con todos los términos en vocabulario: {len(q2)}")
    print(f"[INFO] Queries |q|=3 con todos los términos en vocabulario: {len(q3)}")

    if not q2 and not q3:
        print("[WARN] No hay queries válidas para ejecutar.")
        sys.exit(0)

    algebra = boolean.BooleanAlgebra()

    # ── Ejecutar queries |q|=2 ────────────────────────────────────────────────
    print(f"\n── Ejecutando queries |q|=2 ({len(q2)} queries × {len(PATTERNS_2)} patrones)...")
    results_2 = run_queries(q2, PATTERNS_2, vocabulary, all_docids, store_freq, load_fn, algebra)

    # ── Ejecutar queries |q|=3 ────────────────────────────────────────────────
    print(f"── Ejecutando queries |q|=3 ({len(q3)} queries × {len(PATTERNS_3)} patrones)...")
    results_3 = run_queries(q3, PATTERNS_3, vocabulary, all_docids, store_freq, load_fn, algebra)

    # ── Imprimir resultados ───────────────────────────────────────────────────
    print_results(results_2, results_3, mode_label)

    # ── Guardar archivo de datos ──────────────────────────────────────────────
    out_path = "./output/query_results.csv"
    write_results_file(results_2, results_3, out_path)


if __name__ == "__main__":
    main()
