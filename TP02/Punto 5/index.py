import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

from tokenizer import tokenize  # type: ignore[import]

MIN_TERM_LEN = 2
MAX_TERM_LEN = 25


def is_valid_term(term: str, token_type: str, stopwords: Set[str]) -> bool:
    """
    Filtra un término por longitud y stopwords.

    URLs, emails y nombres propios quedan exceptuados del límite máximo
    porque pueden tener más caracteres que una palabra común.
    """
    if token_type in ("url", "email", "proper"):
        if len(term) < MIN_TERM_LEN:
            return False
    else:
        if len(term) < MIN_TERM_LEN or len(term) > MAX_TERM_LEN:
            return False

    return term not in stopwords


def normalize_tokens(text: str, stopwords: Set[str]) -> List[str]:
    """
    Tokeniza un texto y devuelve sólo términos válidos.

    El tokenizer devuelve pares (token, tipo). Luego se aplican filtros
    de longitud y palabras vacías.
    """
    raw_tokens = tokenize(text)

    valid_tokens = [
        term
        for term, token_type in raw_tokens
        if is_valid_term(term, token_type, stopwords)
    ]

    return valid_tokens


class VectorialIndex:
    """
    Estructura de datos en memoria para recuperación vectorial.

    Guarda:
        documents        {doc_id: nombre_archivo}
        doc_term_freqs   {doc_id: Counter(término -> frecuencia)}
        df               {término: document_frequency}
        doc_weights      {doc_id: {término: peso_tfidf}}
        doc_norms        {doc_id: norma del vector documento}
        inverted_index   {término: {doc_id: peso_tfidf}}
    """

    def __init__(self, stopwords: Set[str]):
        self.stopwords = stopwords

        self.N = 0
        self.documents: Dict[int, str] = {}
        self.doc_term_freqs: Dict[int, Counter] = {}
        self.df: Dict[str, int] = defaultdict(int)

        self.doc_weights: Dict[int, Dict[str, float]] = {}
        self.doc_norms: Dict[int, float] = {}

        self.inverted_index: Dict[str, Dict[int, float]] = defaultdict(dict)

    def index_collection(self, dir_path: str) -> None:
        """Lee documentos .txt del directorio y subdirectorios, y arma las estructuras base."""
        all_files = []
        for root, dirs, files in os.walk(dir_path):
            for f in files:
                if f.endswith(".txt") or f.endswith(".html"):
                    all_files.append(os.path.join(root, f))
        
        all_files.sort()

        for doc_id, path in enumerate(all_files):
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()

            terms = normalize_tokens(content, self.stopwords)
            freqs = Counter(terms)

            rel_path = os.path.relpath(path, dir_path)
            self.documents[doc_id] = rel_path
            self.doc_term_freqs[doc_id] = freqs

            # DF: cada término suma una sola vez por documento.
            for term in freqs.keys():
                self.df[term] += 1

        self.N = len(self.documents)

        if self.N > 0:
            self._build_weights()

    def _idf(self, term: str) -> float:
        """
        Calcula IDF según MIR:

            idf_i = log(N / n_i)

        Se usa logaritmo natural, que es el log por defecto de math.log().
        """
        ni = self.df.get(term, 0)

        if ni == 0:
            return 0.0

        return math.log(self.N / ni)

    def _tf_weight(self, freq: int) -> float:
        """
        Calcula el peso TF según MIR:

            tf = 1 + log(freq)

        Si freq es cero, el peso es cero.
        """
        if freq <= 0:
            return 0.0

        return 1.0 + math.log(freq)

    def _build_weights(self) -> None:
        """Calcula pesos TF/IDF, normas e índice invertido."""
        for doc_id, freqs in self.doc_term_freqs.items():
            weights = {}

            for term, freq in freqs.items():
                wij = self._tf_weight(freq) * self._idf(term)

                # Si idf = 0, el término no aporta discriminación.
                if wij > 0:
                    weights[term] = wij
                    self.inverted_index[term][doc_id] = wij

            self.doc_weights[doc_id] = weights
            self.doc_norms[doc_id] = math.sqrt(sum(w * w for w in weights.values()))

    def build_query_vector(self, query: str) -> Tuple[Dict[str, float], float]:
        """
        Construye el vector TF/IDF de la consulta.

        Para la consulta se aplica el mismo criterio:

            wiq = (1 + log(freqi,q)) * log(N / ni)

        Si un término de la consulta no aparece en la colección, no aporta score.
        """
        query_terms = normalize_tokens(query, self.stopwords)
        query_freqs = Counter(query_terms)

        query_weights = {}

        for term, freq in query_freqs.items():
            if term not in self.df:
                continue

            wiq = self._tf_weight(freq) * self._idf(term)

            if wiq > 0:
                query_weights[term] = wiq

        query_norm = math.sqrt(sum(w * w for w in query_weights.values()))

        return query_weights, query_norm

    def search(self, query: str, max_ranking: int) -> List[Tuple[str, float, float]]:
        """
        Ejecuta una búsqueda usando modelo vectorial.

        Retorna:
            [(nombre_archivo, score_coseno, producto_escalar), ...]
        """
        query_weights, query_norm = self.build_query_vector(query)

        if not query_weights or query_norm == 0:
            return []

        dot_products = defaultdict(float)

        # Producto escalar entre consulta y documentos candidatos.
        for term, wq in query_weights.items():
            postings = self.inverted_index.get(term, {})

            for doc_id, wd in postings.items():
                dot_products[doc_id] += wq * wd

        ranking = []

        # Similitud coseno = producto escalar / (norma query * norma doc).
        for doc_id, dot_product in dot_products.items():
            doc_norm = self.doc_norms.get(doc_id, 0.0)

            if doc_norm == 0:
                continue

            cosine = dot_product / (query_norm * doc_norm)

            if cosine > 0:
                ranking.append((self.documents[doc_id], cosine, dot_product))

        ranking.sort(key=lambda x: x[1], reverse=True)

        return ranking[:max_ranking]
