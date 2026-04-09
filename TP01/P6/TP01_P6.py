#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP01 - Punto 6

Uso:
    python TP01_P6.py <data_dir>

Parámetros:
    data_dir        Directorio de datos (ej: ../data/languageIdentificationData)

Ejemplo:
(parado en la carpeta del punto, y asumiendo que se dejan las colecciones dentro de la carpeta 'data' en la raiz)
    python TP01_P6.py ../data/languageIdentificationData
"""

import os
import re
import sys
import math
import time
import unicodedata
from collections import defaultdict
from os.path import join, isfile, isdir

try:
    import langdetect
except ImportError:
    print("[WARN] El módulo 'langdetect' no está instalado. Instalelo para probar la librería estándar.")
    langdetect = None

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
LANGUAGES = ["English", "French", "Italian"]
LANG_MAP = {
    'en': 'English',
    'fr': 'French',
    'it': 'Italian'
}

def clean_text(text: str) -> str:
    """Filtra y normaliza el texto. Conserva sólo caracteres alfabéticos (a-z)."""
    # Eliminar acentos
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    # Letras a minúsculas, remover puntuación y espacios
    text = re.sub(r'[^a-z]', '', text.lower())
    return text

def train_unigram_model(corpus: str) -> dict:
    """Calcula las probabilidades P(c|Lang) usando suavizado de Laplace (Add-1)."""
    counts = defaultdict(int)
    for c in corpus:
        counts[c] += 1
        
    total_chars = len(corpus)
    vocab_size = len(ALPHABET)
    probs = {}
    
    for char in ALPHABET:
        probs[char] = (counts.get(char, 0) + 1.0) / (total_chars + vocab_size)
    return probs

def train_bigram_model(corpus: str) -> dict:
    """Calcula las probabilidades condicionales P(y|x, Lang) con suavizado de Laplace."""
    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    
    for i in range(len(corpus) - 1):
        x = corpus[i]
        y = corpus[i+1]
        counts[x][y] += 1
        totals[x] += 1
        
    vocab_size = len(ALPHABET)
    probs = defaultdict(dict)
    
    for x in ALPHABET:
        t_x = totals.get(x, 0)
        for y in ALPHABET:
            probs[x][y] = (counts[x].get(y, 0) + 1.0) / (t_x + vocab_size)
    return probs

def score_unigram(text: str, model: dict) -> float:
    """Devuelve el log-score acumulado del modelo Unigrama dado un texto limpio."""
    score = 0.0
    for char in text:
        if char in model:
            score += math.log(model[char])
    return score

def score_bigram(text: str, model: dict) -> float:
    """Devuelve el log-score acumulado del modelo Bigrama dado un texto limpio."""
    score = 0.0
    for i in range(len(text) - 1):
        x = text[i]
        y = text[i+1]
        if x in model and y in model[x]:
            score += math.log(model[x][y])
    return score

def main() -> None:
    data_dir = "../data/languageIdentificationData"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        
    if not isdir(data_dir):
        print(f"[ERROR] El directorio '{data_dir}' no existe.")
        return

    # 1. Fase de Entrenamiento
    print(f"\n[INFO] Entrenando modelos desde: {join(data_dir, 'training')}")
    unigram_models = {}
    bigram_models = {}
    
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
    print("[INFO] Modelos de Unigrama y Bigrama entrenados correctamente.")

    # 2. Carga de Conjunto de Prueba y Soluciones
    test_path = join(data_dir, "test")
    sol_path = join(data_dir, "solution")
    
    if not isfile(test_path) or not isfile(sol_path):
        print(f"[ERROR] Archivos 'test' o 'solution' no encontrados en {data_dir}.")
        return

    with open(test_path, "r", encoding="utf-8", errors="ignore") as f:
        test_lines = f.read().splitlines()

    with open(sol_path, "r", encoding="utf-8", errors="ignore") as f:
        sol_lines = f.read().splitlines()

    solutions = []
    for line in sol_lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            solutions.append(parts[1])  # '1 English' -> 'English'

    if len(solutions) != len(test_lines):
        print("[WARN] Diferente cantidad de líneas en 'test' y 'solution'. Los resultados podrían ser inconsistentes.")

    # 3. Fase de Evaluación
    correct_unigram = 0
    correct_bigram = 0
    correct_langdetect = 0
    total = len(test_lines)

    print(f"\n[INFO] Evaluando {total} documentos de prueba...")
    
    for i, line in enumerate(test_lines):
        truth = solutions[i] if i < len(solutions) else None
        
        # Limpieza de texto para M1 y M2
        cl_text = clean_text(line)
        
        # Saltamos líneas muy vacías si es necesario (el modelo base puede evaluar textos vacíos pero no es util)
        if len(cl_text) == 0:
            continue
        
        # Predicción Unigramas (Maximizando probabilidad condicional)
        best_lang_uni = max(LANGUAGES, key=lambda lang: score_unigram(cl_text, unigram_models[lang]))
        if best_lang_uni == truth:
            correct_unigram += 1
            
        # Predicción Bigramas
        best_lang_bi = max(LANGUAGES, key=lambda lang: score_bigram(cl_text, bigram_models[lang]))
        if best_lang_bi == truth:
            correct_bigram += 1
            
        # Predicción LangDetect
        if langdetect is not None:
            try:
                # langdetect maneja su propia normalización, pasamos el string original (quizá sin vacíos al inicio)
                lang_code = langdetect.detect(line)
                mapped_lang = LANG_MAP.get(lang_code, "Unknown")
                if mapped_lang == truth:
                    correct_langdetect += 1
            except Exception:
                pass # langdetect lanza exception si el texto es muy corto o no tiene features

    # 4. Resultados
    acc_unigram = (correct_unigram / total) * 100 if total > 0 else 0
    acc_bigram = (correct_bigram / total) * 100 if total > 0 else 0
    acc_langdetect = (correct_langdetect / total) * 100 if total > 0 else 0

    print("\n" + "="*45)
    print("      RESULTADOS DE IDENTIFICACIÓN")
    print("="*45)
    print(f"1. Frecuencia de Letras (Unigrama) : {acc_unigram:.2f}%")
    print(f"2. Matriz de Combinación (Bigrama) : {acc_bigram:.2f}%")
    if langdetect is not None:
        print(f"3. Librería externa (langdetect)   : {acc_langdetect:.2f}%")
    else:
        print("3. Librería externa (langdetect)   : NO EVALUADA")
    print("="*45)

if __name__ == "__main__":
    main()