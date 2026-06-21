#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP05 - Punto 5: Recolección intra-dominio + PageRank/Authorities (HITS) con
NetworkX, y comparación de estrategias de crawling (BFS vs orden PageRank)
midiendo el overlap contra el orden por Authority.

Uso:
    python TP05_P5.py <URL> [--max-pages 500]

Ejemplo:
    python TP05_P5.py https://www.unlu.edu.ar
    python TP05_P5.py https://en.wikipedia.org --max-pages 500

Salida:
    output/pages.csv     url, orden_bfs, prof_fisica, pagerank, authority.
    output/overlap.csv    k, overlap_pagerank, overlap_bfs (en %).
    output/overlap.png    evolución del % de overlap vs Authority.
"""

import os
import csv
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Límites y concurrencia ────────────────────────────────────────────────────
MAX_LOGICAL = 100          # tope de seguridad; el límite real es --max-pages
WORKERS = 8                # bajar a 1 si el sitio bloquea por concurrencia

DYNAMIC_EXTS = {".php", ".asp", ".aspx", ".jsp", ".jspx", ".cgi", ".pl", ".do"}


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades de URL
# ══════════════════════════════════════════════════════════════════════════════

def physical_depth(url):
    path = urlparse(url).path
    return len([seg for seg in path.split("/") if seg])


def get_domain(url):
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def normalize_url(url):
    url, _ = urldefrag(url.strip())
    return url


def is_valid_url(url):
    host = urlparse(url).netloc.split("@")[-1].split(":")[0]
    labels = host.split(".")
    return len(labels) >= 2 and all(labels)


# ══════════════════════════════════════════════════════════════════════════════
# Descarga y parseo
# ══════════════════════════════════════════════════════════════════════════════

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (compatible; TP05-Crawler/1.0)"})


def fetch_page(url, timeout=8):
    """Devuelve (html, status, error). html=None si falla."""
    try:
        resp = SESSION.get(url, timeout=timeout)
    except Exception as exc:
        return None, None, str(exc)
    status = resp.status_code
    if not (200 <= status < 300):
        return None, status, f"HTTP {status}"
    if "text/html" not in resp.headers.get("Content-Type", ""):
        return None, status, "no-HTML"
    return resp.text, status, None


def parse_links(html, base_url):
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
# Crawler BFS intra-dominio con tope de páginas
# ══════════════════════════════════════════════════════════════════════════════

def crawl(base_url, max_pages, max_logical=MAX_LOGICAL, workers=WORKERS):
    """
    Crawl BFS dentro del dominio de `base_url`, hasta `max_pages` páginas
    descargadas con éxito.

    Devuelve:
      - pages_order : lista de URLs en orden de descubrimiento (BFS).
      - edges       : lista de (origen, destino) de enlaces intra-dominio.
    """
    base_domain = get_domain(base_url)
    seen = set()
    pages_order = []
    edges = []

    start = normalize_url(base_url)
    frontier = [start]
    seen.add(start)

    logical_depth = 0
    while frontier and logical_depth <= max_logical and len(pages_order) < max_pages:
        remaining = max_pages - len(pages_order)
        batch = frontier[:remaining]
        carry = frontier[remaining:]          # lo no procesado este nivel no se pierde

        print(f"[L{logical_depth}] descargando {len(batch)} páginas "
              f"(acumuladas: {len(pages_order)})")

        results = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_page, u): u for u in batch}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()

        discovered = []
        for url in batch:
            html, _, err = results.get(url, (None, None, None))
            if html is None:
                continue
            pages_order.append(url)
            for link in parse_links(html, url):
                if get_domain(link) != base_domain:
                    continue
                edges.append((url, link))           # arista intra-dominio
                if link not in seen:
                    seen.add(link)
                    discovered.append(link)

        frontier = carry + discovered
        logical_depth += 1

    return pages_order, edges


# ══════════════════════════════════════════════════════════════════════════════
# Análisis: PageRank, HITS y overlap de estrategias
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(pages_order, edges):
    node_set = set(pages_order)
    G = nx.DiGraph()
    G.add_nodes_from(pages_order)
    for src, dst in edges:
        if src in node_set and dst in node_set and src != dst:
            G.add_edge(src, dst)
    return G


def overlap_curve(order, ref_order):
    """% de coincidencia entre los primeros k de `order` y los primeros k de `ref_order`."""
    n = len(order)
    seen_o, seen_r, res = set(), set(), []
    for k in range(1, n + 1):
        seen_o.add(order[k - 1])
        seen_r.add(ref_order[k - 1])
        res.append(100.0 * len(seen_o & seen_r) / k)
    return res


def analyze(pages_order, edges, outdir):
    G = build_graph(pages_order, edges)
    pr = nx.pagerank(G, alpha=0.85)
    try:
        _, auth = nx.hits(G, max_iter=1000, normalized=True)
    except Exception:
        auth = {n: 0.0 for n in G.nodes()}

    bfs_order = list(pages_order)
    pr_order = sorted(pages_order, key=lambda u: pr[u], reverse=True)
    auth_order = sorted(pages_order, key=lambda u: auth[u], reverse=True)

    ov_pr = overlap_curve(pr_order, auth_order)
    ov_bfs = overlap_curve(bfs_order, auth_order)
    ks = list(range(1, len(pages_order) + 1))

    # pages.csv
    rank_bfs = {u: i for i, u in enumerate(bfs_order)}
    with open(os.path.join(outdir, "pages.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url", "orden_bfs", "prof_fisica", "pagerank", "authority"])
        for u in pr_order:
            w.writerow([u, rank_bfs[u], physical_depth(u),
                        f"{pr[u]:.8f}", f"{auth[u]:.8f}"])

    # overlap.csv
    with open(os.path.join(outdir, "overlap.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "overlap_pagerank", "overlap_bfs"])
        for k, a, b in zip(ks, ov_pr, ov_bfs):
            w.writerow([k, f"{a:.4f}", f"{b:.4f}"])

    # overlap.png
    plt.figure(figsize=(9, 5.5))
    plt.plot(ks, ov_pr, label="Crawling por orden de PageRank", linewidth=1.8)
    plt.plot(ks, ov_bfs, label="Crawling por orden BFS", linewidth=1.8)
    plt.xlabel("Páginas crawleadas (k)")
    plt.ylabel("% overlap con top-k por Authority")
    plt.title("Evolución del overlap vs orden por Authority (HITS)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "overlap.png"), dpi=120)

    print(f"\nNodos: {G.number_of_nodes()} | Aristas: {G.number_of_edges()}")
    print(f"overlap@50  -> PageRank: {ov_pr[min(49,len(ov_pr)-1)]:.1f}%  "
          f"BFS: {ov_bfs[min(49,len(ov_bfs)-1)]:.1f}%")
    print("[OK] output/pages.csv, output/overlap.csv, output/overlap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Crawl intra-dominio + PageRank/HITS + overlap.")
    parser.add_argument("base_url", help="URL semilla del dominio.")
    parser.add_argument("--max-pages", type=int, default=500,
                        help="Máx. páginas a descargar (default 500).")
    args = parser.parse_args()

    print(f"Dominio: {get_domain(args.base_url)} | Máx. páginas: {args.max_pages}\n")

    pages_order, edges = crawl(args.base_url, args.max_pages)
    print(f"\nPáginas descargadas: {len(pages_order)} | enlaces intra-dominio: {len(edges)}")

    os.makedirs("output", exist_ok=True)
    analyze(pages_order, edges, "output")


if __name__ == "__main__":
    main()
