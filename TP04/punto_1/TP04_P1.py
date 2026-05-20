#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP04 - Indexador BSBI (Blocked Sort-Based Indexing)

Uso:
    python TP04_P1.py <dir_collection> [opciones]

Parámetros:
    dir_collection              Directorio con documentos .txt de la colección.
    --stopwords <archivo>       (Opcional) Archivo con palabras vacías.
    --n <int>                   Documentos por bloque antes del volcado (default: 10% del total).
    --freq                      Si se indica, almacena docID+frecuencia (8 bytes/posting).
                                Si no, solo docID (4 bytes/posting).
    --output <dir>              Directorio de salida (default: ./bsbi_output/).
    --no-plot                   No generar gráfico de distribución.

Ejemplo:
    python TP04_P1.py ../data/wiki-small/ --n 500 --freq --stopwords ../data/stopwords/english.txt
    python TP04_P1.py ../data/wiki-small/ --n 1000

Archivos generados (en --output):
    chunks/chunk_N.bin      Bloques parciales ordenados por (term_id, docid).
    index.bin               Índice final mergeado.
    vocabulary.pkl          Vocabulario: {term: [seek, df, term_id]}.
    distribucion.png        Gráfico de distribución de tamaños de posting lists.
    reporte.txt             Estadísticas y overhead del índice.

Formato binario (big-endian):
    Chunks:  3 × 4 bytes por registro → (term_id, docid, freq)  = 12 bytes
    Índice (--freq):   2 × 4 bytes por posting → (docid, freq)   = 8 bytes
    Índice (sin freq): 1 × 4 bytes por posting → (docid,)        = 4 bytes
"""

import os
import sys
import struct
import pickle
import heapq
import time
import shutil
from os import listdir
from os.path import join, isdir, isfile
from collections import defaultdict
# import propios
from tokenizer import tokenize

# ── Constantes de formato binario ──────────────────────────────────────────────
# Chunk: (term_id, docid, freq) — 3 unsigned ints big-endian
CHUNK_FMT       = ">III"          # 12 bytes por registro
CHUNK_REC_SIZE  = struct.calcsize(CHUNK_FMT)  # 12

# Índice con frecuencia: (docid, freq) — 2 unsigned ints
IDX_FMT_FREQ    = ">II"           # 8 bytes por posting
IDX_POST_FREQ   = struct.calcsize(IDX_FMT_FREQ)  # 8

# Índice sin frecuencia: (docid,) — 1 unsigned int
IDX_FMT_NFREQ   = ">I"           # 4 bytes por posting
IDX_POST_NFREQ  = struct.calcsize(IDX_FMT_NFREQ)  # 4

MIN_TERM_LEN = 2
MAX_TERM_LEN = 25

# ══════════════════════════════════════════════════════════════════════════════
# funciones auxiliares
# ══════════════════════════════════════════════════════════════════════════════
def load_stopwords(filepath: str) -> set:
    sw = set()
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                sw.add(w)
    return sw


def is_valid_term(term: str, ttype: str, stopwords: set) -> bool:
    if ttype in ("url", "email", "proper"):
        return len(term) >= MIN_TERM_LEN and term not in stopwords
    return MIN_TERM_LEN <= len(term) <= MAX_TERM_LEN and term not in stopwords

VALID_EXTENSIONS = (".txt", ".html", ".htm")

def collect_txt_files(base_dir: str) -> list:
    """
    Recorre recursivamente *base_dir* y devuelve una lista ordenada de
    tuplas (ruta_relativa, ruta_absoluta) para cada archivo de texto/html encontrado.
    """
    result = []
    for dirpath, _dirnames, filenames in os.walk(base_dir):
        for fname in filenames:
            if fname.endswith(VALID_EXTENSIONS):
                abs_path = join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, base_dir)
                result.append((rel_path, abs_path))
    result.sort(key=lambda t: t[0])  # orden estable por ruta relativa
    return result
    
# ══════════════════════════════════════════════════════════════════════════════
# Lectura/escritura de chunks
# ══════════════════════════════════════════════════════════════════════════════

def write_chunk(path: str, tuples: list) -> None:
    """
    Escribe un bloque ordenado de tuplas (term_id, docid, freq) en binario.
    Formato: big-endian, 3 unsigned ints × 4 bytes = 12 bytes/registro.
    """
    tuples.sort(key=lambda t: (t[0], t[1]))  # orden por (term_id, docid)
    with open(path, "wb") as f:
        for term_id, docid, freq in tuples:
            f.write(struct.pack(CHUNK_FMT, term_id, docid, freq))


def write_index_posting(f, docid: int, freq: int, store_freq: bool) -> int:
    """
    Escribe un posting al archivo de índice.
    Retorna cuántos bytes se escribieron.
    """
    if store_freq:
        f.write(struct.pack(IDX_FMT_FREQ, docid, freq))
        return IDX_POST_FREQ
    else:
        f.write(struct.pack(IDX_FMT_NFREQ, docid))
        return IDX_POST_NFREQ


# ══════════════════════════════════════════════════════════════════════════════
# Clase PostingChunk: cursor sobre un chunk en disco
# ══════════════════════════════════════════════════════════════════════════════

class PostingChunk:
    """
    Cursor de lectura secuencial sobre un archivo chunk binario.
    Cada registro: (term_id, docid, freq) = 12 bytes (CHUNK_FMT).

    Se usa en el merge para iterar sobre todos los registros sin cargar
    el archivo completo en memoria.
    """

    def __init__(self, path: str, chunk_id: int):
        self.path = path
        self.chunk_id = chunk_id
        self._fp = open(path, "rb")
        self._current = None
        self._exhausted = False
        self._advance()  # carga el primer registro

    def _advance(self) -> None:
        raw = self._fp.read(CHUNK_REC_SIZE)
        if len(raw) < CHUNK_REC_SIZE:
            self._current = None
            self._exhausted = True
            self._fp.close()
        else:
            self._current = struct.unpack(CHUNK_FMT, raw)  # (term_id, docid, freq)

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    @property
    def term_id(self) -> int:
        return self._current[0]

    @property
    def docid(self) -> int:
        return self._current[1]

    @property
    def freq(self) -> int:
        return self._current[2]

    def next(self) -> None:
        self._advance()

    def __lt__(self, other):
        # Para usar en heapq: comparar por (term_id, docid)
        return (self._current[0], self._current[1]) < (other._current[0], other._current[1])


# ══════════════════════════════════════════════════════════════════════════════
# BSBI: Fase 1 — Construcción de bloques (chunks)
# ══════════════════════════════════════════════════════════════════════════════

def build_chunks(
    dir_path: str,
    stopwords: set,
    n: int,
    chunk_dir: str,
) -> tuple:
    """
    Fase 1 de BSBI: procesar documentos en bloques de n docs, volcando
    cada bloque a disco ordenado por (term_id, docid).

    Retorna:
        term2id   dict  {término_str: term_id_int}
        doc2file  dict  {docid_int: filename_str}
        n_chunks  int   cantidad de chunks generados
        n_docs    int   total de documentos procesados
        t_elapsed float tiempo en segundos
    """
    txt_files = collect_txt_files(dir_path)
    n_docs = len(txt_files)

    term2id: dict[str, int] = {}
    max_term_id = 0
    doc2file: dict[int, str] = {}

    partial_tuples: list[tuple[int, int, int]] = []
    memory_counter = 0  # documentos acumulados en el bloque actual
    chunk_id = 0
    n_chunks = 0

    t_start = time.perf_counter()

    for docid, (rel_path, abs_path) in enumerate(txt_files, start=1):
        doc2file[docid] = rel_path

        with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()

        # Tokenizar y filtrar
        raw_tokens = tokenize(content)
        valid_terms = [
            term for term, ttype in raw_tokens
            if is_valid_term(term, ttype, stopwords)
        ]

        # Calcular frecuencias del documento
        term_freq: dict[str, int] = defaultdict(int)
        for term in valid_terms:
            term_freq[term] += 1

        # Asignar IDs y acumular tuplas
        for term, freq in term_freq.items():
            if term not in term2id:
                max_term_id += 1
                term2id[term] = max_term_id
            partial_tuples.append((term2id[term], docid, freq))

        memory_counter += 1

        # Límite de bloque alcanzado → volcar chunk a disco
        if memory_counter >= n:
            chunk_path = join(chunk_dir, f"chunk_{chunk_id:05d}.bin")
            write_chunk(chunk_path, partial_tuples)
            partial_tuples = []
            memory_counter = 0
            chunk_id += 1
            n_chunks += 1

    # Flush del último bloque incompleto
    if partial_tuples:
        chunk_path = join(chunk_dir, f"chunk_{chunk_id:05d}.bin")
        write_chunk(chunk_path, partial_tuples)
        n_chunks += 1

    t_elapsed = time.perf_counter() - t_start
    return term2id, doc2file, n_chunks, n_docs, t_elapsed


# ══════════════════════════════════════════════════════════════════════════════
# BSBI: Fase 2 — Merge k-way de chunks → índice final
# ══════════════════════════════════════════════════════════════════════════════

def merge_chunks(
    chunk_dir: str,
    n_chunks: int,
    term2id: dict,
    index_path: str,
    vocab_path: str,
    store_freq: bool,
) -> tuple:
    """
    Fase 2 de BSBI: k-way merge de todos los chunks usando un min-heap.

    El merge produce el índice final ordenado por term_id → docid,
    y construye el vocabulario con offsets de disco.

    Retorna:
        vocabulary  dict  {term_str: [seek, df, term_id]}
        posting_sizes list  [df por término, en orden de aparición]
        t_elapsed   float  tiempo en segundos
    """
    # Construir mapa inverso: term_id → term_str
    id2term = {tid: term for term, tid in term2id.items()}

    # Abrir todos los chunks como cursores
    chunk_files = sorted(
        f for f in listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".bin")
    )
    cursors = [PostingChunk(join(chunk_dir, f), i) for i, f in enumerate(chunk_files)]

    # Min-heap: elementos son (term_id, docid, cursor_idx)
    heap = []
    for i, c in enumerate(cursors):
        if not c.exhausted:
            heapq.heappush(heap, (c.term_id, c.docid, i))

    vocabulary: dict[str, list] = {}
    posting_sizes: list[int] = []

    post_size = IDX_POST_FREQ if store_freq else IDX_POST_NFREQ
    seek = 0  # offset actual en el archivo de índice

    t_start = time.perf_counter()

    with open(index_path, "wb") as idx_f:
        current_term_id = None
        current_postings: list[tuple[int, int]] = []  # [(docid, freq), ...]

        def flush_term(term_id: int, postings: list) -> None:
            nonlocal seek
            if not postings:
                return
            df = len(postings)
            term_str = id2term[term_id]
            vocabulary[term_str] = [seek, df, term_id]
            posting_sizes.append(df)
            for docid, freq in postings:
                write_index_posting(idx_f, docid, freq, store_freq)
            seek += df * post_size

        while heap:
            tid, did, ci = heapq.heappop(heap)
            cursor = cursors[ci]
            freq_val = cursor.freq

            if tid != current_term_id:
                # Nuevo término: guardar el anterior
                if current_term_id is not None:
                    flush_term(current_term_id, current_postings)
                current_term_id = tid
                current_postings = [(did, freq_val)]
            else:
                # Mismo término — comprobar si mismo docid (viene de varios chunks)
                if current_postings and current_postings[-1][0] == did:
                    # Acumular freq del mismo (term_id, docid) de distintos chunks
                    current_postings[-1] = (did, current_postings[-1][1] + freq_val)
                else:
                    current_postings.append((did, freq_val))

            cursor.next()
            if not cursor.exhausted:
                heapq.heappush(heap, (cursor.term_id, cursor.docid, ci))

        # Flush del último término
        if current_term_id is not None:
            flush_term(current_term_id, current_postings)

    t_elapsed = time.perf_counter() - t_start

    # Guardar vocabulario
    with open(vocab_path, "wb") as vf:
        pickle.dump(vocabulary, vf)

    return vocabulary, posting_sizes, t_elapsed


# ══════════════════════════════════════════════════════════════════════════════
# Métricas y reporte
# ══════════════════════════════════════════════════════════════════════════════

def compute_overhead(dir_path: str, index_path: str, vocab_path: str) -> dict:
    """Calcula el overhead del índice respecto a la colección."""
    # Tamaño de la colección (solo .txt, recursivo)
    txt_files = collect_txt_files(dir_path)
    col_size = sum(os.path.getsize(abs_p) for _, abs_p in txt_files)
    idx_size = os.path.getsize(index_path)
    voc_size = os.path.getsize(vocab_path)
    total_idx = idx_size + voc_size

    return {
        "coleccion_bytes": col_size,
        "indice_bytes": idx_size,
        "vocabulario_bytes": voc_size,
        "total_indice_bytes": total_idx,
        "overhead_ratio": total_idx / col_size if col_size > 0 else 0,
    }


def write_report(
    output_dir: str,
    n_param: int,
    n_docs: int,
    n_chunks: int,
    t_index: float,
    t_merge: float,
    vocabulary: dict,
    posting_sizes: list,
    overhead: dict,
    store_freq: bool,
) -> None:
    """Escribe reporte.txt."""

    post_size = IDX_POST_FREQ if store_freq else IDX_POST_NFREQ
    mode_str = "docID+frecuencia (8 bytes/posting)" if store_freq else "solo docID (4 bytes/posting)"

    # ── reporte.txt ──────────────────────────────────────────────────────────
    df_values = posting_sizes if posting_sizes else [0]
    df_sum = sum(df_values)
    df_max = max(df_values)
    df_min = min(df_values)
    df_avg = df_sum / len(df_values) if df_values else 0

    # Percentiles simples
    sorted_df = sorted(df_values)
    p50 = sorted_df[len(sorted_df) // 2]
    p90 = sorted_df[int(len(sorted_df) * 0.9)]
    p99 = sorted_df[int(len(sorted_df) * 0.99)]

    hapax = sum(1 for d in df_values if d == 1)

    col_mb = overhead["coleccion_bytes"] / 1_048_576
    idx_mb = overhead["total_indice_bytes"] / 1_048_576
    ratio = overhead["overhead_ratio"]

    with open(join(output_dir, "reporte.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("REPORTE DE INDEXACIÓN BSBI\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Colección: {n_docs} documentos\n")
        f.write(f"Términos únicos en vocabulario: {len(vocabulary)}\n")
        f.write(f"Total de postings: {df_sum}\n")
        f.write(f"Modo de almacenamiento: {mode_str}\n")
        f.write(f"Bytes por posting: {post_size}\n\n")

        f.write("── Tiempos ──────────────────────────────────\n")
        f.write(f"  Indexación (chunks):  {t_index:.4f} s\n")
        f.write(f"  Merge:                {t_merge:.4f} s\n")
        f.write(f"  Total:                {t_index + t_merge:.4f} s\n\n")

        f.write("── Distribución de posting lists ────────────\n")
        f.write(f"  Min DF:  {df_min}\n")
        f.write(f"  Max DF:  {df_max}\n")
        f.write(f"  Media:   {df_avg:.2f}\n")
        f.write(f"  P50:     {p50}\n")
        f.write(f"  P90:     {p90}\n")
        f.write(f"  P99:     {p99}\n")
        f.write(f"  Hapax legomena (DF=1): {hapax} ({100*hapax/len(df_values):.1f}%)\n\n")

        f.write("── Overhead del índice ───────────────────────\n")
        f.write(f"  Tamaño colección: {col_mb:.2f} MB\n")
        f.write(f"  Tamaño índice (index.bin + vocabulary.pkl): {idx_mb:.2f} MB\n")
        f.write(f"  Overhead ratio: {ratio:.4f}  ({ratio*100:.1f}%)\n")

    print(f"[OK] reporte.txt generado.")


# ══════════════════════════════════════════════════════════════════════════════
# Gráfico de distribución de posting lists
# ══════════════════════════════════════════════════════════════════════════════

def plot_distribution(posting_sizes: list, output_path: str) -> None:
    """
    Genera un gráfico log-log de la distribución de tamaños de posting lists
    (DF = document frequency por término).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import collections
    except ImportError:
        print("[WARN] matplotlib no disponible. Se omite el gráfico.")
        return

    # Frecuencia de cada valor de DF
    counter = collections.Counter(posting_sizes)
    sorted_items = sorted(counter.items())
    df_vals = [k for k, _ in sorted_items]
    counts = [v for _, v in sorted_items]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Distribución de tamaños de posting lists (DF)", fontsize=13)

    # Histograma log-log
    ax1 = axes[0]
    ax1.scatter(df_vals, counts, s=8, alpha=0.6, color="steelblue")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("DF (tamaño de posting list)")
    ax1.set_ylabel("Número de términos con ese DF")
    ax1.set_title("Log-log: DF vs Cantidad de términos")
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)

    # Histograma de DFs (recortado en percentil 99 para visibilidad)
    ax2 = axes[1]
    p99 = sorted(posting_sizes)[int(len(posting_sizes) * 0.99)]
    clipped = [d for d in posting_sizes if d <= p99]
    ax2.hist(clipped, bins=50, color="darkorange", edgecolor="white", alpha=0.8)
    ax2.set_xlabel("DF (tamaño de posting list)")
    ax2.set_ylabel("Número de términos")
    ax2.set_title(f"Histograma (hasta percentil 99: DF ≤ {p99})")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"[OK] distribucion.png generado en {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Parseo de argumentos
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    dir_path = sys.argv[1]
    stopwords_file = None
    n_param = None
    store_freq = False
    output_dir = "bsbi_output"
    do_plot = True

    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--stopwords" and i + 1 < len(sys.argv):
            stopwords_file = sys.argv[i + 1]; i += 2
        elif arg == "--n" and i + 1 < len(sys.argv):
            n_param = int(sys.argv[i + 1]); i += 2
        elif arg == "--freq":
            store_freq = True; i += 1
        elif arg == "--output" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]; i += 2
        elif arg == "--no-plot":
            do_plot = False; i += 1
        else:
            print(f"[WARN] Argumento desconocido ignorado: {arg}")
            i += 1

    return dir_path, stopwords_file, n_param, store_freq, output_dir, do_plot


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    dir_path, stopwords_file, n_param, store_freq, output_dir, do_plot = parse_args()

    # Validar directorio de colección
    if not isdir(dir_path):
        print(f"[ERROR] '{dir_path}' no es un directorio válido.")
        sys.exit(1)

    # Contar documentos para definir n por defecto (10% de la colección)
    all_files = collect_txt_files(dir_path)
    total_docs = len(all_files)
    if total_docs == 0:
        print("[ERROR] No se encontraron archivos .txt/.html en el directorio (ni en subdirectorios).")
        sys.exit(1)

    if n_param is None:
        n_param = max(1, total_docs // 10)
        print(f"[INFO] --n no especificado. Usando n = {n_param} (10% de {total_docs} docs).")

    # Cargar stopwords
    stopwords: set = set()
    if stopwords_file:
        if not isfile(stopwords_file):
            print(f"[ERROR] Archivo de stopwords '{stopwords_file}' no encontrado.")
            sys.exit(1)
        stopwords = load_stopwords(stopwords_file)
        print(f"[INFO] Stopwords cargadas: {len(stopwords)} palabras.")

    # Crear directorios de salida
    chunk_dir = join(output_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    index_path = join(output_dir, "index.bin")
    vocab_path = join(output_dir, "vocabulary.pkl")

    mode_str = "docID+frecuencia" if store_freq else "solo docID"
    print(f"\n[INFO] Colección: {total_docs} documentos en '{dir_path}'")
    print(f"[INFO] Modo: {mode_str}")
    print(f"[INFO] n (docs/bloque): {n_param}")
    print(f"[INFO] Salida: {output_dir}\n")

    # ── Fase 1: Construcción de chunks ────────────────────────────────────────
    print("── Fase 1: Construyendo chunks... ──────────────────────────────")
    term2id, doc2file, n_chunks, n_docs, t_index = build_chunks(
        dir_path, stopwords, n_param, chunk_dir
    )
    print(f"[OK] {n_chunks} chunks generados en {t_index:.4f} s")
    print(f"[OK] Vocabulario parcial: {len(term2id)} términos únicos")

    # ── Fase 2: Merge k-way ───────────────────────────────────────────────────
    print("\n── Fase 2: Mergeando chunks... ─────────────────────────────────")
    vocabulary, posting_sizes, t_merge = merge_chunks(
        chunk_dir, n_chunks, term2id, index_path, vocab_path, store_freq
    )
    print(f"[OK] Merge completado en {t_merge:.4f} s")
    print(f"[OK] Vocabulario final: {len(vocabulary)} términos")
    print(f"[OK] Total postings: {sum(posting_sizes)}")

    # ── Métricas y reporte ────────────────────────────────────────────────────
    print("\n── Generando reportes... ───────────────────────────────────────")
    overhead = compute_overhead(dir_path, index_path, vocab_path)
    write_report(
        output_dir, n_param, n_docs, n_chunks,
        t_index, t_merge, vocabulary, posting_sizes,
        overhead, store_freq
    )

    # ── Gráfico ───────────────────────────────────────────────────────────────
    if do_plot and posting_sizes:
        plot_path = join(output_dir, "distribucion.png")
        plot_distribution(posting_sizes, plot_path)

    # ── Tiempos (pantalla) ────────────────────────────────────────────────────
    mode_str = "docID+frecuencia" if store_freq else "solo docID"
    print(f"\n{'=' * 60}")
    print(f"TIEMPOS DE EJECUCIÓN")
    print(f"{'=' * 60}")
    print(f"  Parámetro n (docs/bloque): {n_param}")
    print(f"  Documentos procesados:     {n_docs}")
    print(f"  Chunks generados:          {n_chunks}")
    print(f"  Modo:                      {mode_str}")
    print(f"  Tamaño colección:          {overhead['coleccion_bytes']/1e6:.2f} MB")
    print(f"  Tamaño índice:             {overhead['total_indice_bytes']/1e6:.2f} MB")
    print(f"  Overhead ratio:            {overhead['overhead_ratio']:.4f}  ({overhead['overhead_ratio']*100:.1f}%)")
    print(f"  Tiempo indexación (chunks): {t_index:.4f} s")
    print(f"  Tiempo merge:               {t_merge:.4f} s")
    print(f"  Tiempo total:               {t_index + t_merge:.4f} s")
    print(f"\n[INFO] Todo generado en: {os.path.abspath(output_dir)}")



if __name__ == "__main__":
    main()
