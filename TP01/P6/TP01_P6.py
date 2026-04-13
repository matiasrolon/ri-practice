#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP01 - Punto 6

Uso:
    python TP01_P6.py <data_dir>

Parámetros:
    data_dir        Directorio de datos (ej: ../data/languageIdentificationData)
                    donde están los archivos training, test y solution.

Ejemplo:
    python TP01_P6.py ../data/languageIdentificationData
"""

import re
import sys
import math
import unicodedata
from collections import defaultdict
from os.path import join, isfile, isdir

try:
    import langdetect
except ImportError:
    print("[WARN] El módulo 'langdetect' no está instalado. Instálelo para probar la librería externa.")
    langdetect = None


# Alfabeto ampliado para no perder información útil en francés e italiano.
# Se incluyen letras básicas y caracteres acentuados / especiales frecuentes.
ALPHABET = "abcdefghijklmnopqrstuvwxyzàâäçèéêëîïôöùúûüÿæœìíòó"
LANGUAGES = ["English", "French", "Italian"]
LANG_MAP = {
    "en": "English",
    "fr": "French",
    "it": "Italian"
}


def strip_combining_marks(text: str) -> str:
    """
    Elimina marcas combinantes Unicode sueltas que puedan aparecer
    luego de una normalización inconsistente.
    """
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def clean_text(text: str) -> str:
    """
    Filtra y normaliza el texto.
    Conserva:
      - letras minúsculas a-z
      - letras acentuadas frecuentes en FR/IT
      - caracteres especiales latinos relevantes: ç, æ, œ

    No elimina acentos.
    """
    text = text.lower()
    text = strip_combining_marks(text)

    # Mantener solamente letras presentes en el alfabeto definido
    allowed = set(ALPHABET)
    text = "".join(ch for ch in text if ch in allowed)
    return text


def train_unigram_model(corpus: str) -> dict[str, float]:
    """
    Calcula las probabilidades P(c|Lang) usando suavizado de Laplace (Add-1).
    """
    counts = defaultdict(int)
    for c in corpus:
        counts[c] += 1

    total_chars = len(corpus)
    vocab_size = len(ALPHABET)
    probs: dict[str, float] = {}

    for char in ALPHABET:
        probs[char] = (counts.get(char, 0) + 1.0) / (total_chars + vocab_size)

    return probs


def train_bigram_model(corpus: str) -> dict[str, dict[str, float]]:
    """
    Calcula las probabilidades condicionales P(y|x, Lang)
    con suavizado de Laplace.
    """
    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    for i in range(len(corpus) - 1):
        x = corpus[i]
        y = corpus[i + 1]
        counts[x][y] += 1
        totals[x] += 1

    vocab_size = len(ALPHABET)
    probs: dict[str, dict[str, float]] = defaultdict(dict)

    for x in ALPHABET:
        t_x = totals.get(x, 0)
        for y in ALPHABET:
            probs[x][y] = (counts[x].get(y, 0) + 1.0) / (t_x + vocab_size)

    return probs


def score_unigram(text: str, model: dict[str, float]) -> float:
    """
    Devuelve el log-score acumulado del modelo unigrama dado un texto limpio.
    """
    score = 0.0
    for char in text:
        if char in model:
            score += math.log(model[char])
    return score


def score_bigram(text: str, model: dict[str, dict[str, float]]) -> float:
    """
    Devuelve el log-score acumulado del modelo bigrama dado un texto limpio.
    """
    score = 0.0
    for i in range(len(text) - 1):
        x = text[i]
        y = text[i + 1]
        if x in model and y in model[x]:
            score += math.log(model[x][y])
    return score


def print_error_summary(
    method_name: str,
    correct_count: int,
    total_count: int,
    errors_by_truth: dict[str, int],
    confusion: dict[str, dict[str, int]]
) -> None:
    """
    Imprime un resumen de accuracy, errores por idioma real
    y una matriz de confusión simple.
    """
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"RESUMEN - {method_name}")
    print("=" * 60)
    print(f"Aciertos : {correct_count}/{total_count} ({accuracy:.2f}%)")
    print("Errores por idioma real:")
    for lang in LANGUAGES:
        print(f"  - {lang:<8}: {errors_by_truth[lang]}")

    print("\nMatriz de confusión (idioma real -> idioma predicho):")
    header = "Real\\Pred".ljust(12) + "".join(pred.ljust(12) for pred in LANGUAGES)
    print(header)
    print("-" * len(header))
    for real_lang in LANGUAGES:
        row = real_lang.ljust(12)
        for pred_lang in LANGUAGES:
            row += str(confusion[real_lang][pred_lang]).ljust(12)
        print(row)


def main() -> None:
    data_dir = "../data/languageIdentificationData"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    if not isdir(data_dir):
        print(f"[ERROR] El directorio '{data_dir}' no existe.")
        return

    # 1. Entrenamiento
    print(f"\n[INFO] Entrenando modelos desde: {join(data_dir, 'training')}")
    unigram_models: dict[str, dict[str, float]] = {}
    bigram_models: dict[str, dict[str, dict[str, float]]] = {}

    for lang in LANGUAGES:
        lang_path = join(data_dir, "training", lang)
        if not isfile(lang_path):
            print(f"[ERROR] Archivo de entrenamiento '{lang}' no encontrado en {lang_path}.")
            return

        with open(lang_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        cleaned = clean_text(raw_text)
        unigram_models[lang] = train_unigram_model(cleaned)
        bigram_models[lang] = train_bigram_model(cleaned)

    print("[INFO] Modelos de unigrama y bigrama entrenados correctamente.")

    # 2. Carga de test y soluciones
    test_path = join(data_dir, "test")
    sol_path = join(data_dir, "solution")

    if not isfile(test_path) or not isfile(sol_path):
        print(f"[ERROR] Archivos 'test' o 'solution' no encontrados en {data_dir}.")
        return

    with open(test_path, "r", encoding="utf-8", errors="ignore") as f:
        test_lines = f.read().splitlines()

    with open(sol_path, "r", encoding="utf-8", errors="ignore") as f:
        sol_lines = f.read().splitlines()

    solutions: list[str] = []
    for line in sol_lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            solutions.append(parts[1])

    if len(solutions) != len(test_lines):
        print("[WARN] Distinta cantidad de líneas en 'test' y 'solution'.")
        print("[WARN] Los resultados podrían ser inconsistentes.")

    total = min(len(test_lines), len(solutions))
    print(f"\n[INFO] Evaluando {total} documentos de prueba...")

    # 3. Contadores globales
    correct_unigram = 0
    correct_bigram = 0
    correct_langdetect = 0

    errors_unigram_by_truth = defaultdict(int)
    errors_bigram_by_truth = defaultdict(int)
    errors_langdetect_by_truth = defaultdict(int)

    confusion_unigram = defaultdict(lambda: defaultdict(int))
    confusion_bigram = defaultdict(lambda: defaultdict(int))
    confusion_langdetect = defaultdict(lambda: defaultdict(int))

    skipped_empty = 0
    skipped_langdetect = 0

    # 4. Evaluación
    for i in range(total):
        line = test_lines[i]
        truth = solutions[i]

        cl_text = clean_text(line)

        # Si el texto queda vacío tras la limpieza, no aporta información
        if len(cl_text) == 0:
            skipped_empty += 1
            continue

        # ---- Unigrama
        best_lang_uni = max(
            LANGUAGES,
            key=lambda lang: score_unigram(cl_text, unigram_models[lang])
        )
        confusion_unigram[truth][best_lang_uni] += 1
        if best_lang_uni == truth:
            correct_unigram += 1
        else:
            errors_unigram_by_truth[truth] += 1

        # ---- Bigrama
        best_lang_bi = max(
            LANGUAGES,
            key=lambda lang: score_bigram(cl_text, bigram_models[lang])
        )
        confusion_bigram[truth][best_lang_bi] += 1
        if best_lang_bi == truth:
            correct_bigram += 1
        else:
            errors_bigram_by_truth[truth] += 1

        # ---- Langdetect
        if langdetect is not None:
            try:
                lang_code = langdetect.detect(line)
                mapped_lang = LANG_MAP.get(lang_code, "Unknown")

                if mapped_lang in LANGUAGES:
                    confusion_langdetect[truth][mapped_lang] += 1
                    if mapped_lang == truth:
                        correct_langdetect += 1
                    else:
                        errors_langdetect_by_truth[truth] += 1
                else:
                    skipped_langdetect += 1

            except Exception:
                skipped_langdetect += 1

    evaluated_total = total - skipped_empty

    # 5. Resultados generales
    acc_unigram = (correct_unigram / evaluated_total) * 100 if evaluated_total > 0 else 0.0
    acc_bigram = (correct_bigram / evaluated_total) * 100 if evaluated_total > 0 else 0.0
    acc_langdetect = (correct_langdetect / evaluated_total) * 100 if evaluated_total > 0 else 0.0

    print("\n" + "=" * 50)
    print("RESULTADOS DE IDENTIFICACIÓN")
    print("=" * 50)
    print(f"Documentos evaluados                  : {evaluated_total}")
    print(f"1. Frecuencia de letras (Unigrama)    : {acc_unigram:.2f}%")
    print(f"2. Matriz de combinación (Bigrama)    : {acc_bigram:.2f}%")
    if langdetect is not None:
        print(f"3. Librería externa (langdetect)      : {acc_langdetect:.2f}%")
        print(f"   Casos no evaluados por langdetect  : {skipped_langdetect}")
    else:
        print("3. Librería externa (langdetect)      : NO EVALUADA")
    print("=" * 50)

    # 6. Resúmenes por método
    print_error_summary(
        "Modelo de Unigrama",
        correct_unigram,
        evaluated_total,
        errors_unigram_by_truth,
        confusion_unigram
    )

    print_error_summary(
        "Modelo de Bigrama",
        correct_bigram,
        evaluated_total,
        errors_bigram_by_truth,
        confusion_bigram
    )

    if langdetect is not None:
        print_error_summary(
            "Librería langdetect",
            correct_langdetect,
            evaluated_total,
            errors_langdetect_by_truth,
            confusion_langdetect
        )


if __name__ == "__main__":
    main()