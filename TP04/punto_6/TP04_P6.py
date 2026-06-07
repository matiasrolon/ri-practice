#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP04 - Punto 6: TAAT vs DAAT sobre Dump10K

Uso:
    python TP04_P6.py <dump_file> <queries_file>

Parámetros:
    dump_file       Archivo dump10k.txt (formato term:df:docid1,docid2,...).
    queries_file    Archivo queriesDump10K.txt (una query por línea).

Ejemplo:
    python TP04_P6.py dump10k.txt queriesDump10K.txt

Compara TAAT vs DAAT
Para queries AND. Separa el análisis por longitud de query y tamaño de posting lists.
"""

import os
import sys
import time
from os.path import isfile
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════════════════
# Carga del índice (dump10k.txt)
# ══════════════════════════════════════════════════════════════════════════════

def load_dump_index(filepath):
    """
    Carga dump10k.txt como índice en memoria.
    Formato: term:df:docid1,docid2,...
    Retorna {term: sorted_list_of_docids}.
    """
    index = {}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) < 3:
                continue
            term = parts[0]
            docids_str = parts[2].rstrip(",")
            if not docids_str:
                continue
            docids = sorted(int(d) for d in docids_str.split(",") if d)
            index[term] = docids
    return index


# ══════════════════════════════════════════════════════════════════════════════
# Queries
# ══════════════════════════════════════════════════════════════════════════════

# Carga de queries
def load_queries(filepath, index):
    """
    Lee queries y las agrupa por cantidad de términos.
    Solo incluye queries donde TODOS los términos existen en el índice.
    Retorna {n_terms: [([t1, t2, ...], total_postings), ...]}.
    """
    grouped = defaultdict(list)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            terms = line.strip().lower().split()
            if not terms:
                continue
            if all(t in index for t in terms):
                tp = sum(len(index[t]) for t in terms)
                grouped[len(terms)].append((terms, tp))
    return grouped


# TAAT: intersección AND usando sets
def intersect_taat(index, terms):
    sets = [set(index[t]) for t in terms]
    sets.sort(key=len)
    result = sets[0]
    for i in range(1, len(sets)):
        result = result & sets[i]
        if not result:
            break
    return result


# DAAT: intersección AND con merge sobre listas ordenadas
def intersect_two_daat(a, b):
    result = []
    i, j = 0, 0
    la, lb = len(a), len(b)
    while i < la and j < lb:
        if a[i] == b[j]:
            result.append(a[i])
            i += 1; j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return result


def intersect_daat(index, terms):
    lists = sorted([index[t] for t in terms], key=len)
    result = lists[0]
    for i in range(1, len(lists)):
        result = intersect_two_daat(result, lists[i])
        if not result:
            break
    return result


def run_queries(grouped, index):
    """
    Ejecuta TAAT y DAAT para cada grupo de queries.
    Retorna {n_terms: [(terms, total_postings, t_taat, t_daat, n_results), ...]}.
    """
    results = {}
    for n_terms in sorted(grouped.keys()):
        queries = grouped[n_terms]
        data = []
        for terms, tp in queries:
            if n_terms == 1:
                # Single term: no hay intersección, solo medir la carga
                t0 = time.perf_counter()
                res_taat = set(index[terms[0]])
                t_taat = time.perf_counter() - t0

                t0 = time.perf_counter()
                res_daat = index[terms[0]][:]
                t_daat = time.perf_counter() - t0

                data.append((terms, tp, t_taat, t_daat, len(res_taat)))
            else:
                t0 = time.perf_counter()
                res_taat = intersect_taat(index, terms)
                t_taat = time.perf_counter() - t0

                t0 = time.perf_counter()
                res_daat = intersect_daat(index, terms)
                t_daat = time.perf_counter() - t0

                data.append((terms, tp, t_taat, t_daat, len(res_daat)))

        results[n_terms] = data
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Salida
# ══════════════════════════════════════════════════════════════════════════════

# Análisis por tamaño de posting lists
def classify_by_posting_size(data):
    """
    Clasifica resultados en 3 rangos de tamaño de posting total.
    Retorna {rango_str: [(terms, tp, t_taat, t_daat, n_res), ...]}.
    """
    if not data:
        return {}
    all_tp = sorted(d[1] for d in data)
    p33 = all_tp[len(all_tp) // 3]
    p66 = all_tp[2 * len(all_tp) // 3]

    buckets = {
        f"chico (≤{p33})": [],
        f"medio ({p33+1}–{p66})": [],
        f"grande (>{p66})": [],
    }
    keys = list(buckets.keys())
    for d in data:
        if d[1] <= p33:
            buckets[keys[0]].append(d)
        elif d[1] <= p66:
            buckets[keys[1]].append(d)
        else:
            buckets[keys[2]].append(d)
    return buckets


def print_stats(label, data):
    if not data:
        return
    t_taat = [d[2] for d in data]
    t_daat = [d[3] for d in data]
    s_taat = sum(t_taat)
    s_daat = sum(t_daat)
    n = len(data)
    speedup = s_taat / s_daat if s_daat > 0 else float("inf")

    print(f"    {label} ({n} queries)")
    print(f"      TAAT  total: {s_taat*1000:.4f} ms  prom: {s_taat/n*1000:.4f} ms  min: {min(t_taat)*1000:.4f} ms  max: {max(t_taat)*1000:.4f} ms")
    print(f"      DAAT  total: {s_daat*1000:.4f} ms  prom: {s_daat/n*1000:.4f} ms  min: {min(t_daat)*1000:.4f} ms  max: {max(t_daat)*1000:.4f} ms")
    print(f"      DAAT/TAAT mejora rendimiento: {speedup:.2f}x")
    print()


def print_results(results):
    all_data = []
    for n_terms in sorted(results.keys()):
        data = results[n_terms]
        all_data.extend(data)
        print(f"\n{'=' * 65}")
        print(f"  Queries |q| = {n_terms}  ({len(data)} queries)")
        print(f"{'=' * 65}")

        # Global
        print_stats("Global", data)

    # Análisis general por tamaño de posting
    print(f"\n{'=' * 65}")
    print(f"  Análisis General por Tamaño de Posting Lists  ({len(all_data)} queries)")
    print(f"{'=' * 65}")
    
    buckets = classify_by_posting_size(all_data)
    for rango, bucket_data in buckets.items():
        print_stats(rango, bucket_data)


def write_results_csv(results, output_path):
    """
    Escribe CSV con columnas: query,cantidad_terminos,total_postings,daat,taat
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("query,cantidad_terminos,total_postings,daat,taat\n")
        for n_terms in sorted(results.keys()):
            for terms, tp, t_taat, t_daat, _ in results[n_terms]:
                query_str = " ".join(terms)
                f.write(f"{query_str},{n_terms},{tp},{t_daat:.9f},{t_taat:.9f}\n")
    print(f"[OK] {os.path.basename(output_path)} generado ({output_path}).")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(0)

    dump_file    = sys.argv[1]
    queries_file = sys.argv[2]

    for f in [dump_file, queries_file]:
        if not isfile(f):
            print(f"[ERROR] '{f}' no encontrado.")
            sys.exit(1)

    # Cargar índice
    print("[INFO] Cargando índice desde dump...")
    index = load_dump_index(dump_file)
    total_postings = sum(len(v) for v in index.values())
    print(f"[OK] {len(index)} términos, {total_postings} postings totales.")

    # Cargar y filtrar queries
    grouped = load_queries(queries_file, index)
    for n in sorted(grouped.keys()):
        print(f"[INFO] Queries |q|={n} con todos los términos en vocabulario: {len(grouped[n])}")

    # Ejecutar
    print("\n[INFO] Ejecutando TAAT vs DAAT...")
    results = run_queries(grouped, index)

    # Resultados
    print_results(results)

    # Guardar CSV
    write_results_csv(results, "./output/query_results_p6.csv")


if __name__ == "__main__":
    main()
