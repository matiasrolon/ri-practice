#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descarga una página HTML y extrae sus enlaces.

Uso:
    python TP05_P1.py <URL>

Ejemplo:
    python TP05_P1.py https://unlu.edu.ar
"""

import sys
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


def download_html(url):
    """Descarga el HTML de la URL dada y devuelve el texto de la respuesta."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LinkExtractor/1.0)"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    return response.text


def extract_links(html, base_url):
    """Parsea el HTML y devuelve la lista de enlaces (absolutos) encontrados."""
    soup = BeautifulSoup(html, "html.parser")
    enlaces = []

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        url_absoluta = urljoin(base_url, href)
        enlaces.append(url_absoluta)

    return enlaces


def main():
    if len(sys.argv) != 2:
        print("[ERROR] Debe ingresar una URL como argumento.")
        sys.exit(1)

    url = sys.argv[1]

    try:
        print(f"Descargando: {url}")
        html = download_html(url)
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] No se pudo descargar la página: {e}")
        sys.exit(1)

    enlaces = extract_links(html, url)

    print(f"\nSe encontraron {len(enlaces)} enlaces:\n")
    for enlace in enlaces:
        print(enlace)


if __name__ == "__main__":
    main()