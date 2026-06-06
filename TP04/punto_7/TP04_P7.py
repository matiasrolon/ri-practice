#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP04 - Punto 7: Compresión del índice con VByte (docIDs) y Elias-gamma (freqs)

Uso:
    python TP04_P7.py <index_dir> [--dgaps]

Parámetros:
    index_dir       Directorio del índice (vocabulary.pkl, index.bin).
    --dgaps         Usar delta-gaps en los docIDs antes de comprimir.

Ejemplo:
    python TP04_P7.py ./index
    python TP04_P7.py ./index --dgaps

Archivos generados (en ./output/):
    docids_vbyte.bin            DocIDs comprimidos con Variable-Byte.
    freqs_gamma.bin             Frecuencias comprimidas con Elias-gamma.
    vocabulary_compressed.pkl   {term: [seek, df, tid, vbyte_off, vbyte_len, gamma_off, gamma_len]}
"""

import os
import sys
import struct
import pickle
import time
import math
from os.path import join, isdir, isfile

IDX_FMT  = ">II"
IDX_POST = struct.calcsize(IDX_FMT)


def read_posting_list(index_path, seek, df):
    with open(index_path, "rb") as f:
        f.seek(seek)
        raw = f.read(df * IDX_POST)
    postings = []
    for i in range(df):
        docid, freq = struct.unpack_from(IDX_FMT, raw, i * IDX_POST)
        postings.append((docid, freq))
    return postings

# ══════════════════════════════════════════════════════════════════════════════
# Variable-Byte
# ══════════════════════════════════════════════════════════════════════════════

def vbyte_encode(number):
    # Caso especial: el 0 se codifica como un solo byte con stop bit
    if number == 0:
        return bytes([128]) # 128 = 0b10000000 → datos=0, stop=1
    buf = []
    # Mientras el número no entre en 7 bits (>=128):
    while number >= 128:
        buf.append(number & 0x7F) # mascara AND con 01111111 (guardar 7 bits menos significativos)
        number >>= 7             # descarta esos 7 bits, se queda con el resto
    # El último grupo (cabe en 7 bits): marcar con stop bit (0x80)
    buf.append(number | 0x80)  # mascara OR con 10000000  
    return bytes(buf)


def vbyte_encode_list(numbers):
    return b"".join(vbyte_encode(n) for n in numbers)


# ══════════════════════════════════════════════════════════════════════════════
# Elias-gamma
# ══════════════════════════════════════════════════════════════════════════════

def bits_to_bytes(bit_string):
    """Convierte un string de '0' y '1' a bytes, rellenando con ceros al final."""
    # Rellenar para que sea múltiplo de 8
    while len(bit_string) % 8 != 0:
        bit_string += "0"
    # Agrupar de a 8 bits y convertir cada grupo a un byte
    result = bytearray()
    for i in range(0, len(bit_string), 8):
        byte = int(bit_string[i:i+8], 2)  # "01010011" → entero
        result.append(byte)
    return bytes(result)


def gamma_encode_list(numbers):
    bits = ""
    for k in numbers:
        if k < 1:
            k = 1
        # kd = piso de log2(k) → cantidad de bits para representar k (sin el leading 1)
        kd = int(math.log2(k))
        # kr = k - 2^kd → offset (el número sin su bit más significativo)
        kr = k - (2 ** kd)
        # Parte unaria: kd ceros seguidos de un 1 (kd+1 bits en total)
        bits += "0" * kd + "1"
        # Parte binaria: kr en binario, usando exactamente kd bits
        if kd > 0:
            bits += format(kr, f"0{kd}b")
    return bits_to_bytes(bits)

# ══════════════════════════════════════════════════════════════════════════════
# Dgaps
# ══════════════════════════════════════════════════════════════════════════════

def docids_to_dgaps(docids):
    gaps = [docids[0]]
    for i in range(1, len(docids)):
        gaps.append(docids[i] - docids[i - 1])
    return gaps


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    index_dir = sys.argv[1]
    use_dgaps = "--dgaps" in sys.argv

    if not isdir(index_dir):
        print(f"[ERROR] '{index_dir}' no es un directorio válido.")
        sys.exit(1)

    index_path = join(index_dir, "index.bin")
    vocab_path = join(index_dir, "vocabulary.pkl")

    for p, n in [(index_path, "index.bin"), (vocab_path, "vocabulary.pkl")]:
        if not isfile(p):
            print(f"[ERROR] '{n}' no encontrado en '{index_dir}'.")
            sys.exit(1)

    with open(vocab_path, "rb") as f:
        vocabulary = pickle.load(f)

    orig_size = os.path.getsize(index_path)
    vocab_compressed = {}

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    vbyte_path = join(output_dir, "docids_vbyte.bin")
    gamma_path = join(output_dir, "freqs_gamma.bin")

    t_start = time.perf_counter()

    with open(vbyte_path, "wb") as vf, open(gamma_path, "wb") as gf:
        for term, entry in vocabulary.items():
            seek, df, tid = entry[0], entry[1], entry[2]

            postings = read_posting_list(index_path, seek, df)
            docids = [p[0] for p in postings]
            freqs  = [p[1] for p in postings]

            if use_dgaps:
                docids = docids_to_dgaps(docids)

            vbyte_offset = vf.tell()
            vbyte_data = vbyte_encode_list(docids)
            vf.write(vbyte_data)

            gamma_offset = gf.tell()
            gamma_data = gamma_encode_list(freqs)
            gf.write(gamma_data)

            vocab_compressed[term] = [seek, df, tid, vbyte_offset, len(vbyte_data), gamma_offset, len(gamma_data)]

    t_elapsed = time.perf_counter() - t_start

    vc_path = join(output_dir, "vocabulary_compressed.pkl")
    with open(vc_path, "wb") as f:
        pickle.dump(vocab_compressed, f)

    vbyte_size = os.path.getsize(vbyte_path)
    gamma_size = os.path.getsize(gamma_path)
    comp_total = vbyte_size + gamma_size
    dgaps_str = "con DGaps" if use_dgaps else "sin DGaps"

    print(f"\n── Compresión ({dgaps_str}) ─────────────────────────────")
    print(f"  index.bin original:     {orig_size:>12,} bytes  ({orig_size/1e6:.2f} MB)")
    print(f"  docids_vbyte.bin:       {vbyte_size:>12,} bytes  ({vbyte_size/1e6:.2f} MB)")
    print(f"  freqs_gamma.bin:        {gamma_size:>12,} bytes  ({gamma_size/1e6:.2f} MB)")
    print(f"  Total comprimido:       {comp_total:>12,} bytes  ({comp_total/1e6:.2f} MB)")
    print(f"  Ratio:                  {comp_total/orig_size*100:.1f}%")
    print(f"  Tiempo de compresión:   {t_elapsed:.4f} s")


if __name__ == "__main__":
    main()
