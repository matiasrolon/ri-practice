#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP04 - Punto 7.1: Recuperar posting list original y comprimida de un término

Uso:
    python TP04_P7_1.py <index_dir> <compressed_dir> <termino> [--dgaps]

Parámetros:
    index_dir       Directorio del índice original (index.bin, vocabulary.pkl).
    compressed_dir  Directorio de archivos comprimidos (docids_vbyte.bin,
                    freqs_gamma.bin, vocabulary_compressed.pkl).
    termino         Término a buscar.
    --dgaps         Indicar si el índice comprimido fue creado con delta-gaps.

Ejemplo:
    python TP04_P7_1.py ./index ./output "dragon"
    python TP04_P7_1.py ./index ./output "cowboy" --dgaps

Salida:
    Posting list original (desde index.bin) y descomprimida (desde archivos
    comprimidos), con tiempos de lectura/descompresión.
"""

import sys
import struct
import pickle
import time
import math
from os.path import join, isdir, isfile

IDX_FMT  = ">II"
IDX_POST = struct.calcsize(IDX_FMT)


# ══════════════════════════════════════════════════════════════════════════════
# VByte decode
# ══════════════════════════════════════════════════════════════════════════════

def vbyte_decode_list(data, count):
    result = []
    pos = 0
    for _ in range(count):
        n = 0
        shift = 0
        while True:
            b = data[pos]
            pos += 1
            n |= (b & 0x7F) << shift
            shift += 7
            if b & 0x80:
                break
        result.append(n)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Elias-gamma decode
# ══════════════════════════════════════════════════════════════════════════════

def bytes_to_bits(data):
    """Convierte bytes a un string de '0' y '1'."""
    return "".join(format(byte, "08b") for byte in data)


def gamma_decode_list(data, count):
    bits = bytes_to_bits(data)
    pos = 0
    result = []
    for _ in range(count):
        # Leer parte unaria: contar ceros hasta encontrar un 1 → eso es kd
        kd = 0
        while bits[pos] == "0":
            kd += 1
            pos += 1
        pos += 1  # saltar el "1" que cierra la parte unaria

        if kd == 0:
            # kd=0 → el número es 1 (no hay parte binaria)
            result.append(1)
        else:
            # Leer parte binaria: los siguientes kd bits son kr
            kr = int(bits[pos:pos + kd], 2)
            pos += kd
            # Reconstruir: k = 2^kd + kr
            k = (2 ** kd) + kr
            result.append(k)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades
# ══════════════════════════════════════════════════════════════════════════════

def dgaps_to_docids(gaps):
    docids = [gaps[0]]
    for i in range(1, len(gaps)):
        docids.append(docids[-1] + gaps[i])
    return docids


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(0)

    index_dir      = sys.argv[1]
    compressed_dir = sys.argv[2]
    term           = sys.argv[3].strip().lower()
    use_dgaps      = "--dgaps" in sys.argv

    if not isdir(index_dir):
        print(f"[ERROR] '{index_dir}' no es un directorio válido.")
        sys.exit(1)
    if not isdir(compressed_dir):
        print(f"[ERROR] '{compressed_dir}' no es un directorio válido.")
        sys.exit(1)

    index_path = join(index_dir, "index.bin")
    vocab_path = join(index_dir, "vocabulary.pkl")
    doc2file_path = join(index_dir, "doc2file.pkl")
    vbyte_path = join(compressed_dir, "docids_vbyte.bin")
    gamma_path = join(compressed_dir, "freqs_gamma.bin")
    vc_path    = join(compressed_dir, "vocabulary_compressed.pkl")

    for p, n, d in [(vocab_path, "vocabulary.pkl", index_dir),
                    (index_path, "index.bin", index_dir),
                    (doc2file_path, "doc2file.pkl", index_dir),
                    (vc_path, "vocabulary_compressed.pkl", compressed_dir),
                    (vbyte_path, "docids_vbyte.bin", compressed_dir),
                    (gamma_path, "freqs_gamma.bin", compressed_dir)]:
        if not isfile(p):
            print(f"[ERROR] '{n}' no encontrado en '{d}'.")
            sys.exit(1)

    with open(vocab_path, "rb") as f:
        vocabulary = pickle.load(f)
    with open(vc_path, "rb") as f:
        vocab_compressed = pickle.load(f)
    with open(doc2file_path, "rb") as f:
        doc2file = pickle.load(f)

    if term not in vocabulary:
        print(f"[ERROR] Término '{term}' no existe en el vocabulario.")
        sys.exit(1)

    seek, df, tid = vocabulary[term][0], vocabulary[term][1], vocabulary[term][2]
    _, _, _, vb_off, vb_len, gm_off, gm_len = vocab_compressed[term]

    # ── Lectura original (index.bin) ──────────────────────────────────────────
    t0 = time.perf_counter()
    with open(index_path, "rb") as f:
        f.seek(seek)
        raw = f.read(df * IDX_POST)
    orig_docids = []
    orig_freqs = []
    for i in range(df):
        docid, freq = struct.unpack_from(IDX_FMT, raw, i * IDX_POST)
        orig_docids.append(docid)
        orig_freqs.append(freq)
    t_orig = time.perf_counter() - t0

    # ── Descompresión (VByte + Gamma) ─────────────────────────────────────────
    t0 = time.perf_counter()

    with open(vbyte_path, "rb") as f:
        f.seek(vb_off)
        vb_data = f.read(vb_len)

    with open(gamma_path, "rb") as f:
        f.seek(gm_off)
        gm_data = f.read(gm_len)

    dec_docids = vbyte_decode_list(vb_data, df)
    dec_freqs  = gamma_decode_list(gm_data, df)

    if use_dgaps:
        dec_docids = dgaps_to_docids(dec_docids)

    t_decomp = time.perf_counter() - t0

    # ── Verificación ──────────────────────────────────────────────────────────
    match = (dec_docids == orig_docids and dec_freqs == orig_freqs)

    # ── Salida ────────────────────────────────────────────────────────────────
    dgaps_str = "con DGaps" if use_dgaps else "sin DGaps"
    print(f"\nTérmino: '{term}'  (df={df}, {dgaps_str})")
    print(f"  Lectura original (index.bin):  {df * IDX_POST} bytes  {t_orig*1000:.4f} ms")
    print(f"  Lectura comprimida (VByte+γ):  {vb_len + gm_len} bytes  {t_decomp*1000:.4f} ms")
    print(f"  Ratio compresión:              {(vb_len + gm_len) / (df * IDX_POST) * 100:.1f}%")
    print(f"  Verificación:                  {'OK' if match else 'ERROR'}")

    print(f"\n  Posting list original:")
    print(f"  {'DocName':<50} {'docID':>7}  {'Frecuencia':>10}")
    print(f"  " + "-" * 72)
    for d, fr in zip(orig_docids, orig_freqs):
        doc_name = doc2file.get(d, "<desconocido>")
        print(f"  {doc_name:<50} {d:>7}  {fr:>10}")
    print(f"  " + "-" * 72)

    print(f"\n  Versión comprimida en disco:")
    print(f"    VByte ({vb_len} bytes): {vb_data.hex()}")
    print(f"    Gamma ({gm_len} bytes): {gm_data.hex()}")


if __name__ == "__main__":
    main()
