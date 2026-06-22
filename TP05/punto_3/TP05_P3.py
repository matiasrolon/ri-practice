#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP05 - Punto 3: Crawling de un único dominio para análisis de enlaces

Uso:
    python TP05_P3.py <URL> [--max-pages N]

Parametros:
    <URL>             URL base/semilla del dominio a indexar.
    --max-pages       (Opcional) Cantidad maxima de paginas a descargar.

Ejemplo:
    python TP05_P3.py https://www.ebay.com
    python TP05_P3.py https://en.wikipedia.org --max-pages 100

Salida:
    output/pages.csv   url, prof_logica, prof_fisica, tipo (dinamica/estatica).
"""

import os
import csv
import argparse
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

# ── Constantes ────────────────────────────────────────────────────────────────
MAX_LOGICAL = 4
MAX_PHYSICAL = 4
WORKERS = 12

# Extensiones por las que se asume una página generada dinamicamente.
DYNAMIC_EXTS = {".php", ".asp", ".aspx", ".jsp", ".jspx", ".cgi", ".pl", ".do"}


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades
# ══════════════════════════════════════════════════════════════════════════════

def physical_depth(url):
    """Profundidad física = niveles de directorio en el path. '/' -> 0, '/a/b' -> 2."""
    path = urlparse(url).path
    return len([seg for seg in path.split("/") if seg])


def get_domain(url):
    """Dominio (netloc) sin 'www.'."""
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def normalize_url(url):
    """Quita el fragmento (#...) y espacios."""
    url, _ = urldefrag(url.strip())
    return url


def is_valid_url(url):
    """True si el host está bien formado (sin labels vacíos, p.ej. 'x..com')."""
    host = urlparse(url).netloc.split("@")[-1].split(":")[0]
    labels = host.split(".")
    return len(labels) >= 2 and all(labels)


def classify_url(url):
    """
    Clasifica la URL como 'dinamica' si tiene query string o extensión de script; 
    'estatica' en caso contrario.
    """
    parsed = urlparse(url)
    if parsed.query:
        return "dinamica"
    ext = os.path.splitext(parsed.path)[1].lower()
    if ext in DYNAMIC_EXTS:
        return "dinamica"
    return "estatica"


# ══════════════════════════════════════════════════════════════════════════════
# Descarga y parseo
# ══════════════════════════════════════════════════════════════════════════════

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (compatible; TP05-Crawler/1.0)"})


def fetch_page(url, timeout=8):
    """
    Descarga la página. Devuelve una tupla (html, status_code, error_msg)
    """
    try:
        resp = SESSION.get(url, timeout=timeout)
    except Exception as exc:
        return None, None, str(exc)

    status = resp.status_code
    if not (200 <= status < 300):
        return None, status, f"HTTP {status}"

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        return None, status, f"Content-Type: {content_type}"

    return resp.text, status, None


def parse_links(html, base_url):
    """Enlaces absolutos http/https con host válido, sin anclas ni javascript."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        absolute = normalize_url(urljoin(base_url, href))
        if (absolute.startswith("http://") or absolute.startswith("https://")) \
                and is_valid_url(absolute):
            links.append(absolute)
    return links


# ══════════════════════════════════════════════════════════════════════════════
# Filtro de URLs (solo mismo dominio + límites)
# ══════════════════════════════════════════════════════════════════════════════

def passes_filter(url, base_domain, max_physical):
    """Filtra por dominio y profundidad física."""
    if get_domain(url) != base_domain:
        return False
    if physical_depth(url) > max_physical:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Crawler
# ══════════════════════════════════════════════════════════════════════════════

def crawl(base_url, max_logical, max_physical, workers=8, max_pages=None):
    """
    Devuelve `pages`: lista de (url, prof_logica, prof_fisica, tipo) por cada
    página descargada dentro del dominio.
    Si max_pages es None, no hay límite de páginas.
    """
    base_domain = get_domain(base_url)
    seen = set()
    pages_per_site = defaultdict(int)   # cuenta paginas por dominio
    pages = []

    start = normalize_url(base_url)
    frontier = [start]
    seen.add(start)

    logical_depth = 0
    while frontier and logical_depth <= max_logical:
        # Construir batch: sin límite si max_pages es None, o respetando el cupo
        if max_pages is None:
            batch = list(frontier)
        else:
            batch = []
            for url in frontier:
                if pages_per_site[get_domain(url)] < max_pages:
                    pages_per_site[get_domain(url)] += 1
                    batch.append(url)

        print(f"[L{logical_depth}] descargando {len(batch)} páginas...")

        results = {}
        ok_count = 0
        fail_count = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_page, u): u for u in batch}
            for fut in as_completed(futures):
                url = futures[fut]
                html, status, err = fut.result()
                results[url] = (html, status, err)
                if html is not None:
                    ok_count += 1
                    print(f"  [OK]   {url}")
                else:
                    fail_count += 1
                    print(f"  [FAIL] {url}  →  {err}")
        print(f"  Resultado: {ok_count} OK, {fail_count} fallidas")

        next_frontier = []
        links_total   = 0
        links_new     = 0
        links_same_domain = 0
        links_filtered    = 0
        for url in batch:
            html, _, _ = results.get(url, (None, None, None))
            if html is None:
                continue
            # registrar la página descargada para el análisis
            pages.append((url, logical_depth, physical_depth(url), classify_url(url)))

            raw_links = parse_links(html, url)
            links_total += len(raw_links)
            for link in raw_links:
                # Si pertenece al mismo dominio
                if get_domain(link) == base_domain:
                    links_same_domain += 1
                # Si cumple maximo nivel logica y no está en las paginas ya vistas.
                if logical_depth >= max_logical or link in seen:
                    continue
                # Si cumple maximo nivel fisico y no cambia de dominio.
                if passes_filter(link, base_domain, max_physical):
                    next_frontier.append(link)
                    seen.add(link)
                    links_new += 1
                else:
                    links_filtered += 1

        print(f"  Links encontrados: {links_total} totales, "
              f"{links_same_domain} del dominio, "
              f"{links_new} encolados, "
              f"{links_filtered} descartados por filtro")

        frontier = next_frontier
        logical_depth += 1

    return pages


# ══════════════════════════════════════════════════════════════════════════════
# Estadisticas de salida
# ══════════════════════════════════════════════════════════════════════════════

def write_pages_csv(pages, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url", "prof_logica", "prof_fisica", "tipo"])
        w.writerows(pages)
    print(f"[OK] Detalle por página: {path}")


def print_distributions(pages):
    tipo = Counter(p[3] for p in pages)
    log = Counter(p[1] for p in pages)
    fis = Counter(p[2] for p in pages)

    print("\n Estadisticas\n")
    for k in ("dinamica", "estatica"):
        print(f"  {k:9s}: {tipo.get(k, 0)}")

    print("\n Frecuencia por profundidad lógica")
    for d in sorted(log):
        if d == 0:
            continue
        print(f"  L{d}: {log[d]}")

    print("\n Frecuencia por profundidad física")
    for d in sorted(fis):
        if d == 0:
            continue
        print(f"  F{d}: {fis[d]}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Crawl de un dominio + análisis.")
    parser.add_argument("base_url",
                        help="URL base/semilla del dominio a indexar.")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="(Opcional) Máx. páginas a descargar")
    args = parser.parse_args()

    print(f"Dominio: {get_domain(args.base_url)}")
    limite_str = str(args.max_pages) if args.max_pages is not None else "sin límite"
    print(f"Máx. páginas: {limite_str} | Lógica máx: {MAX_LOGICAL} | Física máx: {MAX_PHYSICAL}\n")

    pages = crawl(args.base_url, MAX_LOGICAL, MAX_PHYSICAL, WORKERS,
                  max_pages=args.max_pages)

    print(f"\nPáginas descargadas: {len(pages)}")

    os.makedirs("output", exist_ok=True)
    write_pages_csv(pages, os.path.join("output", "pages.csv"))
    print_distributions(pages)


if __name__ == "__main__":
    main()