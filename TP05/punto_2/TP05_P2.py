#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP05 - Punto 2: Crawler básico web + grafo con pyvis.

Uso:
    python TP05_P2.py [--max-pages-site 30]

Parámetros:
    --max-pages-site      Cantidad maxima de paginas a visitar por sitio (default 30) 

Ejemplo:
    python TP05_P2.py --max-pages-site 30

Salida:
    output/crawl_graph.html   Grafo interactivo (pyvis) de páginas → enlaces.
    edges.csv                 Aristas del grafo (origen, destino).

"""

import os
import sys
import csv
import time
import argparse
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup
from pyvis.network import Network

# ── Constantes de límites ─────────────────────────────────────────────────────
MAX_LOGICAL = 3
MAX_PHYSICAL = 3


# ── Conjunto semilla por defecto: top 20 Netcraft (puede actualizarse) ────────
DEFAULT_SEEDS = [
    "https://www.google.com",
    "https://www.youtube.com",
    "https://www.facebook.com",
    "https://www.instagram.com",
    "https://www.x.com",
    "https://www.wikipedia.org",
    "https://www.reddit.com",
    "https://www.amazon.com",
    "https://www.tiktok.com",
    "https://www.linkedin.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.netflix.com",
    "https://www.bing.com",
    "https://www.live.com",
    "https://www.office.com",
    "https://www.yahoo.com",
    "https://www.whatsapp.com",
    "https://www.openai.com",
    "https://www.github.com",
]


# ══════════════════════════════════════════════════════════════════════════════
# Funciones de profundidad
# ══════════════════════════════════════════════════════════════════════════════

def physical_depth(url):
    """
    Profundidad física = número de niveles de directorio en el path de la URL.

    Ejemplos:
        https://sitio.com/            -> 0
        https://sitio.com/a           -> 1
        https://sitio.com/a/b         -> 2
    """
    path = urlparse(url).path
    # quitar slash inicial/final y contar segmentos no vacíos
    segments = [seg for seg in path.split("/") if seg]
    return len(segments)


def get_domain(url):
    """Devuelve el dominio (netloc) de la URL, sin el 'www.'."""
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def normalize_url(url):
    """Normaliza una URL: quita el fragmento (#...) y espacios."""
    url, _ = urldefrag(url.strip())
    return url


def is_valid_url(url):
    """True si la URL tiene un host bien formado (sin labels vacíos, p.ej. 'x..com')."""
    host = urlparse(url).netloc.split("@")[-1].split(":")[0]
    labels = host.split(".")
    return len(labels) >= 2 and all(labels)


# ══════════════════════════════════════════════════════════════════════════════
# Descarga y parseo
# ══════════════════════════════════════════════════════════════════════════════

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (compatible; TP05-Crawler/1.0)"})


def fetch_page(url, timeout=8):
    """
    Descarga la página. Devuelve el HTML (str) o None si falla
    (error de red, URL malformada, no es HTML, status != 200, etc.).
    """
    try:
        resp = SESSION.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception:
        return None

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        return None

    return resp.text


def parse_links(html, base_url):
    """Extrae los enlaces absolutos de un HTML, ignorando anclas y javascript."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        absolute = normalize_url(urljoin(base_url, href))
        # solo http/https y host bien formado
        if (absolute.startswith("http://") or absolute.startswith("https://")) \
                and is_valid_url(absolute):
            links.append(absolute)
    return links


# ══════════════════════════════════════════════════════════════════════════════
# Filtro de URLs (corresponde al "pass our filter" del algoritmo)
# ══════════════════════════════════════════════════════════════════════════════

def passes_filter(url, max_physical, pages_per_site, max_pages_site):
    """
    Decide si una URL debe agregarse a la todo_list.

    Aplica:
      - Profundidad física máxima.
      - Límite de páginas por sitio (dominio).
    """
    if physical_depth(url) > max_physical:
        return False

    domain = get_domain(url)
    if pages_per_site[domain] >= max_pages_site:
        return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
# Crawler — implementa el algoritmo de la figura con restricciones
# ══════════════════════════════════════════════════════════════════════════════

def crawl(seeds, max_pages_site, max_logical, max_physical, workers=12):
    """
    Crawler BFS (mismo algoritmo de la figura) pero descargando cada nivel
    de profundidad lógica en paralelo con un ThreadPoolExecutor.

    Restricciones:
      - max_logical : profundidad lógica máxima (saltos desde la semilla).
      - max_physical: profundidad física máxima (niveles de directorio).
      - max_pages_site: páginas máximas descargadas por dominio.
    """
    done_list = set()
    seen = set()                           # URLs ya encoladas o procesadas
    pages_per_site = defaultdict(int)
    edges = []

    # frontera inicial: semillas en profundidad lógica 0
    frontier = []
    for seed in seeds:
        seed = normalize_url(seed)
        if seed not in seen:
            frontier.append(seed)
            seen.add(seed)

    logical_depth = 0
    while frontier and logical_depth <= max_logical:
        # tanda de descarga, respetando el límite por sitio
        batch = []
        for url in frontier:
            domain = get_domain(url)
            if pages_per_site[domain] < max_pages_site:
                pages_per_site[domain] += 1
                batch.append(url)

        print(f"[L{logical_depth}] descargando {len(batch)} páginas...")

        # descarga concurrente
        results = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_page, u): u for u in batch}
            for fut in as_completed(futures):
                u = futures[fut]
                done_list.add(u)
                results[u] = fut.result()

        # parseo + construcción de la frontera siguiente
        next_frontier = []
        for url in batch:
            html = results.get(url)
            if html is None:
                continue
            for link in parse_links(html, url):
                edges.append((url, link))             # arista del grafo
                if logical_depth >= max_logical:
                    continue
                if link in seen:
                    continue
                if passes_filter(link, max_physical, pages_per_site, max_pages_site):
                    next_frontier.append(link)
                    seen.add(link)

        frontier = next_frontier
        logical_depth += 1

    return edges, done_list, pages_per_site


# ══════════════════════════════════════════════════════════════════════════════
# Construcción del grafo con pyvis
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(edges, seeds, out_path):
    """Construye un grafo dirigido con pyvis a partir de las aristas."""
    net = Network(height="800px", width="100%", directed=True,
                  bgcolor="#ffffff", font_color="#333333")
    net.barnes_hut()  # layout físico para grafos grandes

    seed_set = {normalize_url(s) for s in seeds}
    nodes = set()

    # Agregar nodos
    for origen, destino in edges:
        nodes.add(origen)
        nodes.add(destino)

    for node in nodes:
        domain = get_domain(node)
        is_seed = node in seed_set
        color = "#E24B4A" if is_seed else "#185FA5"
        size = 25 if is_seed else 12
        # etiqueta corta: solo el dominio
        net.add_node(node, label=domain, title=node, color=color, size=size)

    # Agregar aristas
    for origen, destino in edges:
        net.add_edge(origen, destino)

    net.set_options("""
    {
      "physics": {
        "barnesHut": { "gravitationalConstant": -8000, "springLength": 120 },
        "minVelocity": 0.75
      }
    }
    """)

    net.save_graph(out_path)
    print(f"\n[OK] Grafo guardado en: {out_path}")
    print(f"     Nodos: {len(nodes)}  |  Aristas: {len(edges)}")


def save_edges_csv(edges, path):
    """Guarda las aristas en un CSV (origen, destino)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["origen", "destino"])
        writer.writerows(edges)
    print(f"[OK] Aristas guardadas en: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Crawler básico web + grafo pyvis.")
    parser.add_argument("--max-pages-site", type=int, default=30,
                        help="Máx. páginas por sitio/dominio (default 30).")
    args = parser.parse_args()

    seeds = DEFAULT_SEEDS   # semillas

    print(f"Semillas: {len(seeds)} sitios")
    print(f"Máx. páginas/sitio: {args.max_pages_site}")
    print(f"Profundidad lógica máx: {MAX_LOGICAL}")
    print(f"Profundidad física máx: {MAX_PHYSICAL}\n")

    # Crawl
    edges, done_list, pages_per_site = crawl(
        seeds,
        max_pages_site=args.max_pages_site,
        max_logical=MAX_LOGICAL,
        max_physical=MAX_PHYSICAL,
    )

    print(f"\n── Recolección finalizada ──")
    print(f"Páginas descargadas: {len(done_list)}")
    print(f"Sitios visitados: {len(pages_per_site)}")
    print(f"Aristas (enlaces) registradas: {len(edges)}")

    # Crear la carpeta output si no existe
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    out_path = os.path.join(output_dir, "crawl_graph.html")

    # Grafo + CSV
    build_graph(edges, seeds, out_path)
    save_edges_csv(edges, "edges.csv")


if __name__ == "__main__":
    main()