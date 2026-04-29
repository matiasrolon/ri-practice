#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP - Modelo Vectorial con TF/IDF según MIR

Uso:
    python TP02_P5.py <dir_coleccion> [--stopwords <archivo_stopwords>] [--max-ranking <k>]

Parámetros:
    dir_coleccion              Directorio con documentos .txt.
    --stopwords <archivo>      (Opcional) Archivo con palabras vacías.
                               Por defecto: ../data/english.txt
    --max-ranking <k>          (Opcional) Cantidad máxima de resultados a mostrar.
                               Por defecto: 10.

Ejemplos:
    python TP02_P5.py ../data/wiki-small/
    python TP02_P5.py ../data/wiki-small/ --max-ranking 20
    python TP02_P5.py ../data/wiki-small/ --stopwords ../data/english.txt
    python3 TP02_P5.py ../data/wiki-small/ --stopwords ../data/stopwords/english.txt --max-ranking 50

El programa:
    1. Lee documentos .txt desde un directorio.
    2. Tokeniza los documentos usando tokenizer.py.
    3. Elimina palabras vacías si se proporciona archivo.
    4. Arma una estructura en memoria para recuperación vectorial.
    5. Calcula pesos TF/IDF según MIR: wij = (1 + log(freqi,j)) * log(N / ni)
       donde:
           freqi,j = frecuencia del término i en el documento j
           N       = cantidad total de documentos
           ni      = cantidad de documentos donde aparece el término i

    6. Permite ingresar consultas por consola.
    7. Devuelve ranking por similitud coseno.
"""

import csv
import sys
from os.path import isdir, isfile
from typing import List, Set, Tuple

from index import VectorialIndex, MIN_TERM_LEN, MAX_TERM_LEN

# parametros defult
DEFAULT_STOPWORDS = "../data/stopwords/english.txt"
DEFAULT_MAX_RANKING = 10

def load_stopwords(filepath: str) -> Set[str]:
    """Carga las palabras vacías desde un archivo"""
    stopwords = set()

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                stopwords.add(word)

    return stopwords


def parse_args() -> Tuple[str, str, int]:
    """
    Parsea los argumentos de línea de comandos.

    Retorna:
        dir_path        Directorio de la colección.
        stopwords_file  Archivo de palabras vacías.
        max_ranking     Cantidad máxima de resultados a mostrar.
    """
    if len(sys.argv) < 2:
        print("[ERROR] Debe indicar el directorio de la colección.")
        sys.exit(1)

    dir_path = sys.argv[1]
    stopwords_file = DEFAULT_STOPWORDS
    max_ranking = DEFAULT_MAX_RANKING

    if "--stopwords" in sys.argv:
        idx = sys.argv.index("--stopwords")
        if idx + 1 < len(sys.argv):
            stopwords_file = sys.argv[idx + 1]
        else:
            print("[ERROR] Se indicó --stopwords pero no se proporcionó el archivo.")
            sys.exit(1)

    if "--max-ranking" in sys.argv:
        idx = sys.argv.index("--max-ranking")
        if idx + 1 < len(sys.argv):
            try:
                max_ranking = int(sys.argv[idx + 1])
                if max_ranking <= 0:
                    raise ValueError
            except ValueError:
                print("[ERROR] --max-ranking debe ser un entero positivo.")
                sys.exit(1)
        else:
            print("[ERROR] Se indicó --max-ranking pero no se proporcionó el valor.")
            sys.exit(1)

    return dir_path, stopwords_file, max_ranking


def print_ranking(query: str, ranking: List[Tuple[str, float, float]]) -> None:
    """Imprime el ranking obtenido en consola."""
    print("\n" + "=" * 80)
    print(f"Consulta: {query}")
    print("=" * 80)

    if not ranking:
        print("[INFO] No se encontraron documentos relevantes para la consulta.")
        return

    print(f"{'Pos':>4}  {'Documento':<55}  {'Coseno':>12}")
    print("-" * 80)

    for pos, (filename, cosine, _) in enumerate(ranking, start=1):
        print(f"{pos:>4}  {filename:<55}  {cosine:>12.6f}")


def interactive_search(index: VectorialIndex, max_ranking: int) -> None:
    """Permite ingresar consultas hasta que el usuario decida terminar."""
    while True:
        print("\nIngrese una consulta, o presione ENTER para terminar.")
        query = input("consulta> ").strip()

        if not query:
            print("[INFO] Fin del programa.")
            break

        ranking = index.search(query, max_ranking)
        print_ranking(query, ranking)

        if ranking:
            csv_filename = f"{query.replace(' ', '_')}.csv"
            try:
                with open(csv_filename, mode='w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Pos', 'Documento', 'Coseno'])
                    for pos, (doc, cosine, _) in enumerate(ranking, start=1):
                        writer.writerow([pos, doc, f"{cosine:.6f}"])
                print(f"[INFO] Resultados guardados en '{csv_filename}'.")
            except Exception as e:
                print(f"[ERROR] No se pudo guardar el archivo CSV: {e}")

        answer = input("\n¿Desea realizar otra consulta? [s/N]: ").strip().lower()

        if answer not in ("s", "si", "sí", "y", "yes"):
            print("[INFO] Fin del programa.")
            break


def main() -> None:
    dir_path, stopwords_file, max_ranking = parse_args()

    # Validar directorio de colección.
    if not isdir(dir_path):
        print(f"[ERROR] '{dir_path}' no es un directorio válido.")
        sys.exit(1)

    # Cargar stopwords. Por defecto se intenta usar ../data/english.txt.
    stopwords = set()

    if stopwords_file:
        if isfile(stopwords_file):
            stopwords = load_stopwords(stopwords_file)
            print(f"[INFO] Stopwords cargadas: {len(stopwords)} palabras desde '{stopwords_file}'.")
        else:
            print(f"[WARN] No se encontró el archivo de stopwords '{stopwords_file}'.")
            print("[WARN] Se continuará sin eliminar palabras vacías.")

    print(f"[INFO] Directorio de colección: {dir_path}")
    print(f"[INFO] Ranking máximo: {max_ranking}")
    print(f"[INFO] MIN_TERM_LEN={MIN_TERM_LEN}, MAX_TERM_LEN={MAX_TERM_LEN}")
    print("[INFO] Indexando colección...\n")

    index = VectorialIndex(stopwords)
    index.index_collection(dir_path)

    if index.N == 0:
        print("[WARN] No se encontraron documentos .txt en el directorio.")
        sys.exit(0)

    print("[OK] Indexación finalizada.")
    print(f"[INFO] Documentos indexados: {index.N}")
    print(f"[INFO] Términos únicos: {len(index.df)}")

    interactive_search(index, max_ranking)


if __name__ == "__main__":
    main()
