#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
from os import listdir
from os.path import join, isdir
from tokenizer import tokenize

def main():
    cont = 1
    proccess_docs = 0
    total_tokens_count = 0
    df = {} # Document Frequency
    all_terms = set()
    
    if len(sys.argv) < 2:
        print('El usuario NO ha pasado argumentos suficientes. Se estableceran los valores por defecto')
        dir_path = '../collection_test/TestCollection'
        delete_sw = False
        stop_words = []
    else:    
        dir_path = sys.argv[1]
        
        delete_sw = False
        stop_words = []
        if len(sys.argv) >= 3 and sys.argv[2].lower() == 'true': 
            delete_sw = True
            if len(sys.argv) >= 4:
                path_sw = sys.argv[3]
                if os.path.exists(path_sw):
                    with open(path_sw, "r", encoding="utf-8", errors="ignore") as sw_file:
                        stop_words = [line.strip() for line in sw_file.readlines()]
                else:
                    print(f"No se encontro el archivo de stopwords: {path_sw}")

    print('Directorio a analizar    > ' + dir_path)
    if not delete_sw:
        print('No se Eliminaran palabras vacias.')        
    else:
        print('Se eliminaran palabras vacias.')
    print('---------------------------------------------------')
        
    if isdir(dir_path):
        l = listdir(dir_path)
        for arch in l:
            if arch.endswith('.txt'):   
                path_arch = join(dir_path, arch)
                with open(path_arch, "r", encoding="utf-8", errors="ignore") as file_arch:
                    content = file_arch.read()
                
                # Tokenizamos
                tokens_doc = tokenize(content)
                
                # Borramos stopwords
                if delete_sw:
                    tokens_doc = [t for t in tokens_doc if t not in stop_words]
                
                total_tokens_count += len(tokens_doc)
                
                # Extraemos y contamos tokens únicos para Document Frequency
                terms_in_doc = set(tokens_doc)
                all_terms.update(terms_in_doc)
                
                for term in terms_in_doc:
                    df[term] = df.get(term, 0) + 1
                    
                cont += 1
                proccess_docs += 1
        
        # Resultados pedidos para la consigna
        print(f"\nLista de términos y su DF (Mostrando algunos de ejemplo):")
        # Mostrar 10 como muestra para no llenar la consola si hay muchos
        muestra_df = list(df.items())[:10]
        for v in muestra_df:
            print(f"   {v[0]}: {v[1]}")
        print("   ...")
        
        print(f"\nCantidad de tokens: {total_tokens_count}")
        print(f"Cantidad de términos: {len(all_terms)}")
        print(f"Cantidad de documentos procesados: {proccess_docs}")

        # Comparar con collection_data.json
        print('\n--- COMPARACIÓN CON collection_data.json ---')
        # Buscamos el json que se encuentre en '../collection_test/collection_data.json'
        json_path = join(os.path.dirname(os.path.normpath(dir_path)), "collection_data.json")
        if not os.path.exists(json_path):
            json_path = join(dir_path, '../collection_data.json')
            
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding="utf-8") as f:
                coll_data = json.load(f)
                
            coincidencias = 0
            fallas = 0
            for data in coll_data.get("data", []):
                term = data["term"]
                expected_df = data["df"]
                actual_df = df.get(term, 0)
                
                if expected_df == actual_df:
                    coincidencias += 1
                else:
                    fallas += 1
                    print(f"Falla en el termino '{term}': DF calculado = {actual_df} | DF esperado = {expected_df}")
            
            if fallas == 0:
                print(f"✔️ Todos los {coincidencias} terminos comparados coinciden exitosamente con su respectivo DF.")
            else:
                print(f"Terminos comparados exitosamente: {coincidencias}. Fallas: {fallas}.")
        else:
            print(f"No se pudo localizar el archivo collection_data.json en {json_path}")
            
    else:
        print("El paramero pasado no corresponde a un directorio")

if __name__ == '__main__':
    main()