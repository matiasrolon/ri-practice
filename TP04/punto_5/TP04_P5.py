#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP04 - Punto 5: Consultas AND con y sin Skip Lists

Uso:
    python TP04_P5.py <index_dir> <queries_file> [--freq]

Parámetros:
    index_dir           Directorio del índice (vocabulary.pkl, index.bin, doc2file.pkl).
    queries_file        Archivo de queries (formato "id:texto", una por línea).
    --freq              El índice fue creado con frecuencias (8 bytes/posting).

En la primera ejecución construye skips.bin y extiende vocabulary.pkl con
el campo skip_seek. En ejecuciones posteriores detecta que ya existen.

Archivos generados/modificados (en index_dir):
    skips.bin                Skip pointers binarios (docid, idx) × 8 bytes c/u.
    vocabulary.pkl           Sobreescribe: {term: [seek, df, term_id, skip_seek]}.
    query_results_skip.csv   Resultados de las queries.
"""

import os
import sys
import math
import struct
import pickle
import time
from os.path import join, isdir, isfile


# ── Constantes de formato ─────────────────────────────────────────────────────
IDX_FMT_FREQ   = ">II"
IDX_FMT_NFREQ  = ">I"
IDX_POST_FREQ  = struct.calcsize(IDX_FMT_FREQ)   # 8
IDX_POST_NFREQ = struct.calcsize(IDX_FMT_NFREQ)  # 4

SKIP_FMT  = ">II"                                # (docid, index_in_list)
SKIP_SIZE = struct.calcsize(SKIP_FMT)             # 8


# ══════════════════════════════════════════════════════════════════════════════
# Carga del índice
# ══════════════════════════════════════════════════════════════════════════════

def load_index_metadata(index_dir):
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


def load_posting_list_disk(index_path, seek, df, store_freq):
    post_size = IDX_POST_FREQ if store_freq else IDX_POST_NFREQ
    fmt       = IDX_FMT_FREQ if store_freq else IDX_FMT_NFREQ
    with open(index_path, "rb") as f:
        f.seek(seek)
        raw = f.read(df * post_size)
    docids = []
    offset = 0
    for _ in range(df):
        docids.append(struct.unpack_from(fmt, raw, offset)[0])
        offset += post_size
    return docids


# ══════════════════════════════════════════════════════════════════════════════
# Skip Lists
# ══════════════════════════════════════════════════════════════════════════════

def skip_step_for(df):
    """Paso de skip determinístico a partir del DF."""
    return max(1, int(math.sqrt(df)))


def n_skips_for(df):
    """Cantidad de skip pointers derivada del DF (sin necesidad de almacenarla)."""
    if df <= 1:
        return 0
    step = skip_step_for(df)
    return math.ceil(df / step)


def build_skip_list(postings):
    n = len(postings)
    if n <= 1:
        return []
    step = skip_step_for(n)
    return [(postings[i], i) for i in range(0, n, step)]


def build_and_save_skips(vocabulary, index_path, store_freq, skips_bin_path, vocab_path):
    """
    Construye skip pointers para todos los términos, los escribe en skips.bin,
    y sobreescribe vocabulary.pkl extendiendo cada entrada con skip_seek.
    """
    total = len(vocabulary)

    with open(skips_bin_path, "wb") as sf:
        for i, (term, entry) in enumerate(vocabulary.items(), 1):
            seek, df, tid = entry[0], entry[1], entry[2]
            pl = load_posting_list_disk(index_path, seek, df, store_freq)
            skips = build_skip_list(pl)

            skip_seek = sf.tell()
            for docid, idx in skips:
                sf.write(struct.pack(SKIP_FMT, docid, idx))

            # Extender la entrada del vocabulario
            vocabulary[term] = [seek, df, tid, skip_seek]

            if i % 50000 == 0 or i == total:
                print(f"  [{i}/{total}] skip lists escritas...")

    # Sobreescribir vocabulary.pkl con el campo adicional
    with open(vocab_path, "wb") as f:
        pickle.dump(vocabulary, f)

    bin_size = os.path.getsize(skips_bin_path)
    total_skips = sum(n_skips_for(entry[1]) for entry in vocabulary.values())
    print(f"[OK] skips.bin generado ({bin_size / 1e6:.2f} MB, {total_skips} skip pointers).")
    print(f"[OK] vocabulary.pkl actualizado (ahora incluye skip_seek).")


def load_skip_list_from_disk(skips_bin_path, skip_seek, df):
    """Lee los skip pointers de un término desde skips.bin."""
    n = n_skips_for(df)
    if n == 0:
        return []
    with open(skips_bin_path, "rb") as f:
        f.seek(skip_seek)
        raw = f.read(n * SKIP_SIZE)
    return [struct.unpack_from(SKIP_FMT, raw, i * SKIP_SIZE) for i in range(n)]


def vocab_has_skips(vocabulary):
    """Detecta si el vocabulario ya fue extendido con skip_seek."""
    for entry in vocabulary.values():
        return len(entry) >= 4
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Intersección AND — sin skips
# ══════════════════════════════════════════════════════════════════════════════

def intersect_no_skip(list_a, list_b):
    result = []
    i, j = 0, 0
    len_a, len_b = len(list_a), len(list_b)
    while i < len_a and j < len_b:
        if list_a[i] == list_b[j]:
            result.append(list_a[i])
            i += 1; j += 1
        elif list_a[i] < list_b[j]:
            i += 1
        else:
            j += 1
    return result


def intersect_multi_no_skip(lists):
    lists = sorted(lists, key=len)
    result = lists[0]
    for k in range(1, len(lists)):
        result = intersect_no_skip(result, lists[k])
        if not result:
            break
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Intersección AND — con skips
# ══════════════════════════════════════════════════════════════════════════════

def intersect_with_skip(list_a, skips_a, list_b, skips_b):
    result = []
    i, j = 0, 0
    len_a, len_b = len(list_a), len(list_b)
    si, sj = 0, 0

    while i < len_a and j < len_b:
        if list_a[i] == list_b[j]:
            result.append(list_a[i])
            i += 1; j += 1
        elif list_a[i] < list_b[j]:
            target = list_b[j]
            skipped = False
            while si + 1 < len(skips_a) and skips_a[si + 1][0] <= target:
                si += 1; skipped = True
            if skipped and skips_a[si][1] > i:
                i = skips_a[si][1]
            else:
                i += 1
        else:
            target = list_a[i]
            skipped = False
            while sj + 1 < len(skips_b) and skips_b[sj + 1][0] <= target:
                sj += 1; skipped = True
            if skipped and skips_b[sj][1] > j:
                j = skips_b[sj][1]
            else:
                j += 1
    return result


def intersect_multi_with_skip(lists_and_skips):
    items = sorted(lists_and_skips, key=lambda x: len(x[0]))
    result = items[0][0]
    result_skips = items[0][1]
    for k in range(1, len(items)):
        result = intersect_with_skip(result, result_skips, items[k][0], items[k][1])
        result_skips = build_skip_list(result)
        if not result:
            break
    return result


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
                _, text = line.split(":", 1)
            else:
                text = line
            terms = text.strip().lower().split()
            if terms:
                queries.append(terms)
    return queries


def filter_queries(queries, vocabulary, n_terms):
    return [
        terms for terms in queries
        if len(terms) == n_terms and all(t in vocabulary for t in terms)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Patrones AND
# ══════════════════════════════════════════════════════════════════════════════

PATTERNS_2 = {"t1 AND t2": lambda t: t[:2]}
PATTERNS_3 = {"t1 AND t2 AND t3": lambda t: t[:3]}


def total_posting_size(terms, vocabulary):
    return sum(vocabulary[t][1] for t in terms)


def run_queries(queries, patterns, vocabulary, store_freq, index_path, skips_bin_path):
    results = {p: [] for p in patterns}

    for terms in queries:
        tp = total_posting_size(terms, vocabulary)

        for pat_name, pat_fn in patterns.items():
            q_terms = pat_fn(terms)

            lists = []
            lists_and_skips = []
            for t in q_terms:
                seek, df, _, skip_seek = vocabulary[t]
                pl = load_posting_list_disk(index_path, seek, df, store_freq)
                skips = load_skip_list_from_disk(skips_bin_path, skip_seek, df)
                lists.append(pl)
                lists_and_skips.append((pl, skips))

            t0 = time.perf_counter()
            res_no = intersect_multi_no_skip([l[:] for l in lists])
            t_no = time.perf_counter() - t0

            t0 = time.perf_counter()
            res_sk = intersect_multi_with_skip([(l[:], s) for l, s in lists_and_skips])
            t_sk = time.perf_counter() - t0

            results[pat_name].append((q_terms, tp, t_no, t_sk, len(res_no)))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Salida
# ══════════════════════════════════════════════════════════════════════════════

def print_results(results_2, results_3):
    for label, results in [("Queries |q|=2", results_2), ("Queries |q|=3", results_3)]:
        print(f"\n── {label} ─────────────────────────────────────────────")
        for pat_name, data in results.items():
            if not data:
                print(f"  {pat_name}: sin queries válidas")
                continue

            t_no_list  = [d[2] for d in data]
            t_sk_list  = [d[3] for d in data]
            postings   = [d[1] for d in data]

            for method, times in [("Sin skip", t_no_list), ("Con skip", t_sk_list)]:
                print(f"\n  Patrón: {pat_name} ({method})")
                print(f"    Queries ejecutadas:  {len(data)}")
                print(f"    Tiempo total:        {sum(times):.6f} s")
                print(f"    Tiempo promedio:     {sum(times)/len(times):.6f} s")
                print(f"    Tiempo mínimo:       {min(times):.6f} s")
                print(f"    Tiempo máximo:       {max(times):.6f} s")
                print(f"    Posting total prom:  {sum(postings)/len(postings):.1f}")

            total_no = sum(t_no_list)
            total_sk = sum(t_sk_list)
            speedup = total_no / total_sk if total_sk > 0 else float("inf")
            print(f"\n    Speedup global {pat_name}: {speedup:.2f}x")


def write_results_file(results_2, results_3, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("query,metodo,patron,total_postings,tiempo_seg\n")
        for results in [results_2, results_3]:
            for pat_name, data in results.items():
                for q_terms, tp, t_no, t_sk, _ in data:
                    query_str = " ".join(q_terms)
                    f.write(f"{query_str},sin_skip,{pat_name},{tp},{t_no:.9f}\n")
                    f.write(f"{query_str},con_skip,{pat_name},{tp},{t_sk:.9f}\n")
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

    vocab_path     = join(index_dir, "vocabulary.pkl")
    skips_bin_path  = join(index_dir, "skips.bin")

    # ── Cargar índice ─────────────────────────────────────────────────────────
    vocabulary, doc2file = load_index_metadata(index_dir)
    print(f"[INFO] Vocabulario: {len(vocabulary)} términos, {len(doc2file)} documentos.")
    print(f"[INFO] Modo postings: {'docID+freq' if store_freq else 'solo docID'}")

    # ── Skip lists: construir o detectar que ya existen ───────────────────────
    if vocab_has_skips(vocabulary) and isfile(skips_bin_path):
        bin_size = os.path.getsize(skips_bin_path)
        print(f"[INFO] Skip lists ya presentes (skips.bin {bin_size / 1e6:.2f} MB, vocabulary.pkl con skip_seek).")
    else:
        print(f"\n── Construyendo skip lists para {len(vocabulary)} términos...")
        build_and_save_skips(vocabulary, index_path, store_freq, skips_bin_path, vocab_path)

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

    # ── Ejecutar queries ──────────────────────────────────────────────────────
    print(f"\n── Ejecutando queries |q|=2 ({len(q2)} queries × {len(PATTERNS_2)} patrones)...")
    results_2 = run_queries(q2, PATTERNS_2, vocabulary, store_freq, index_path, skips_bin_path)

    print(f"── Ejecutando queries |q|=3 ({len(q3)} queries × {len(PATTERNS_3)} patrones)...")
    results_3 = run_queries(q3, PATTERNS_3, vocabulary, store_freq, index_path, skips_bin_path)

    # ── Imprimir resultados ───────────────────────────────────────────────────
    print_results(results_2, results_3)

    # ── Guardar CSV ───────────────────────────────────────────────────────────
    out_path = join(index_dir, "query_results_skip.csv")
    write_results_file(results_2, results_3, out_path)


if __name__ == "__main__":
    main()
