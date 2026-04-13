#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP01 - Punto 5

Uso:
    python TP01_P5.py <file_collection> [--stopwords <file>]

Parámetros:
    file_collection           Archivo con los documentos de la colección.
    --stopwords <file>        (Opcional) Archivo con palabras vacías a eliminar.

Ejemplo:
Asumiendo que se dejan las colecciones dentro de 'data' en la carpeta raiz, y una vez parado en la carpeta del punto (/P5)
    python TP01_P5.py ../data/vaswani/corpus/doc-text.trec
    python TP01_P5.py ../data/vaswani/corpus/doc-text.trec --stopwords ../data/stopwords/spanish.txt

"""

import os
import re
import time
import sys
from os.path import isfile, abspath
from nltk.stem import PorterStemmer, LancasterStemmer

def parse_args() -> str:
    # Por defecto apuntamos a doc-text.trec
    file_path = "../data/vaswani/corpus/doc-text.trec"
    if len(sys.argv) < 2:
        print(f"[INFO] No se pasó un archivo por parámetro. Se usará el archivo por defecto: {file_path}")
    else:
        file_path = sys.argv[1]
    return file_path

def process_file(file_path: str) -> list[str]:
    """Extrae el texto del documento o colección y lo tokeniza."""
    word_re = re.compile(r"[a-z]+(?:[-'][a-z]+)*")
    
    all_tokens: list[str] = []
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
        content = fh.read()

    # Extraer contenido de etiquetas <DOC>
    docs = re.findall(r'<DOC>(.*?)</DOC>', content, re.DOTALL)
    
    # Si no hay etiquetas <DOC>, se toma todo el archivo como un único documento
    if not docs:
        docs = [content]
        
    for doc in docs:
        # Remover etiquetas DOCNO
        doc_text = re.sub(r'<DOCNO>.*?</DOCNO>', '', doc)
        doc_text = doc_text.lower()
        # Encontrar tokens con la regex
        tokens = word_re.findall(doc_text)
        all_tokens.extend(tokens)
        
    return all_tokens

def main() -> None:
    file_path = parse_args()
    
    if not isfile(file_path):
        print(f"[ERROR] El archivo '{file_path}' no es válido o no existe.")
        return

    print(f"[INFO] Leyendo archivo desde: {file_path}")
    start_read = time.time()
    tokens = process_file(file_path)
    print(f"[INFO] Se extrajeron {len(tokens)} tokens en total. Tiempo de lectura y tokenización: {time.time()-start_read:.2f}s")
    
    if not tokens:
        print("[WARN] No se extrajeron tokens del archivo. Verifique su contenido y formato.")
        return

    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    
    porter_stems: list[str] = []
    lancaster_stems: list[str] = []
    
    # ---------------------------
    # Ejecución Porter
    # ---------------------------
    print("\n--- Ejecutando Porter Stemmer en toda la colección ---")
    start_porter = time.time()
    for token in tokens:
        porter_stems.append(porter.stem(token))
    time_porter = time.time() - start_porter
    unique_porter = set(porter_stems)
    print(f"Tiempo de ejecución Porter : {time_porter:.4f} segundos")
    print(f"Tokens únicos con Porter  : {len(unique_porter)}")

    # ---------------------------
    # Ejecución Lancaster
    # ---------------------------
    print("\n--- Ejecutando Lancaster Stemmer en toda la colección ---")
    start_lancaster = time.time()
    for token in tokens:
        lancaster_stems.append(lancaster.stem(token))
    time_lancaster = time.time() - start_lancaster
    unique_lancaster = set(lancaster_stems)
    print(f"Tiempo de ejecución Lancaster: {time_lancaster:.4f} segundos")
    print(f"Tokens únicos con Lancaster : {len(unique_lancaster)}")

    # ---------------------------
    # Comparación 1 a 1 
    # ---------------------------
    unique_original_tokens = set(tokens)
    total_unique = len(unique_original_tokens)
    print(f"\n--- Comparación 1 a 1 sobre vocabulario original (Tokens únicos: {total_unique}) ---")
    
    equal_stems: int = 0
    different_stems: int = 0
    
    comparison_mapping: list[str] = []
    
    for token in unique_original_tokens:
        p_stem = porter.stem(token)
        l_stem = lancaster.stem(token)
        
        comparison_mapping.append(f"{token:25} {p_stem:20} {l_stem:20}")
        if p_stem == l_stem:
            equal_stems += 1
        else:
            different_stems += 1
            
    pct_equal = (equal_stems / total_unique) * 100 if total_unique > 0 else 0.0
    pct_different = (different_stems / total_unique) * 100 if total_unique > 0 else 0.0
            
    print(f"Stems resultantes iguales (Porter == Lancaster): {equal_stems} ({pct_equal:.2f}%)")
    print(f"Stems resultantes diferentes (Porter != Lancaster): {different_stems} ({pct_different:.2f}%)")
    
    # Guardar en archivo para visualizar
    out_file = "comparation_1_to_1.txt"
    with open(out_file, "w", encoding="utf-8") as out:
        out.write("Comparación 1 a 1 de tokens de la colección Vaswani:\n")
        out.write(f"Stems iguales: {equal_stems} | Stems diferentes: {different_stems}\n")
        out.write("-" * 90 + "\n")
        out.write(f"{'Token':25} {'Porter':20} {'Lancaster':20}\n")
        out.write("-" * 90 + "\n")
        for line in sorted(comparison_mapping):
            out.write(line + "\n")
            
    print(f"\n[INFO] Archivo de comparación detallada 1 a 1 guardado en: {abspath(out_file)}")

if __name__ == "__main__":
    main()