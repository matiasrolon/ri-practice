#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP01 - Punto 1

Uso:
    python TP01_P1.py <path_dir_collection> <path_collection_stats>

Parámetros:
    path_dir_collection     Path del directorio con los documentos de la colección.
    path_collection_stats   Archivo con las estadísticas de la colección (para comparar con los resultados).

Ejemplo:
(parado en la carpeta del punto, asumiendo que se dejan las colecciones dentro de la carpeta 'data' en la raiz):
    python TP01_P1.py
    python TP01_P1.py ../data/collection_test/TestCollection 
    python TP01_P1.py ../data/collection_test/TestCollection ../data/collection_test/collection_data.json
"""

import os
import sys
import json
from os import listdir
from os.path import join, isdir
from tokenizer import tokenize


def parse_args():
    """Parsea argumentos posicionales. El segundo (stats json) es opcional."""
    dir_path = '../collection_test/TestCollection'
    json_path = None

    if len(sys.argv) < 2:
        print('El usuario NO ha pasado parametros. Se establecerán por defecto.')
    elif len(sys.argv) < 3:
        dir_path = sys.argv[1]
    else:
        dir_path = sys.argv[1]
        json_path = sys.argv[2]

    return dir_path, json_path


def main():
    docs_count = 0
    total_tokens_count = 0
    df = {}  # Document Frequency
    all_terms = set()

    dir_path, json_path = parse_args()
    print('Directorio a analizar: ' + dir_path)

    if isdir(dir_path):
        l = listdir(dir_path)
        for arch in l:
            if arch.endswith('.txt'):   
                path_arch = join(dir_path, arch)
                with open(path_arch, "r", encoding="utf-8", errors="ignore") as file_arch:
                    content = file_arch.read()
                
                # Tokenizamos
                tokens_doc = tokenize(content)
                
                total_tokens_count += len(tokens_doc)
                # Extraemos y contamos tokens únicos para Document Frequency
                terms_in_doc = set(tokens_doc)
                all_terms.update(terms_in_doc)
                # Todos los terminos de este documento, sumaran 1 a su DF
                for term in terms_in_doc:
                    df[term] = df.get(term, 0) + 1
                    
                docs_count += 1
        
        # Resultados
        print(f"\n> Lista de términos y su DF:")
        # Mostrar 10 como muestra para no llenar la consola si hay muchos
        list_df = list(df.items())
        for term, df_count in list_df:
            print(f"{term} {df_count}")
        
        print(f"\n> Cantidad de tokens: {total_tokens_count}")
        print(f"> Cantidad de términos: {len(all_terms)}")
        print(f"> Cantidad de documentos procesados: {docs_count}")

        # -----------------------------------------------------------------------
        # Comparar con collection_data.json (solo si se pasó como argumento)
        if json_path:
            print('\n--- COMPARACIÓN CON collection_data.json ---')
            if not os.path.exists(json_path):
                print(f'> No se pudo localizar el archivo: {json_path}')
            else:
                with open(json_path, 'r', encoding="utf-8") as f:
                    coll_data = json.load(f)

                stats = coll_data.get("statistics", {})
                if stats:
                    print(f"> Cantidad de tokens: {stats.get('num_tokens', 'N/A')}")
                    print(f"> Cantidad de términos: {stats.get('num_terms', 'N/A')}")
                    print(f"> Cantidad de documentos procesados: {stats.get('N', 'N/A')}\n")

                coincidences = 0
                failures = 0
                for data in coll_data.get("data", []):
                    term = data["term"]
                    expected_df = data["df"]
                    current_df = df.get(term, 0)

                    if expected_df == current_df:
                        coincidences += 1
                    else:
                        failures += 1
                        print(f"> Falla en el termino '{term}': DF calculado = {current_df} | DF esperado = {expected_df}")

                if failures == 0:
                    print(f"> Todos los {coincidences} terminos comparados coinciden exitosamente con su respectivo DF.")
                else:
                    print(f"> Terminos comparados exitosamente: {coincidences}. Fallas: {failures}.")
            
    else:
        print("> El paramero pasado no corresponde a un directorio")

if __name__ == '__main__':
    main()