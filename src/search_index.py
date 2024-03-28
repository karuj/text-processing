import json
import math
import heapq
import uuid
from time import time
import tqdm

def get_doc_freq(text, n_grams=3):
    doc_freq = {}
    words = text.lower().split(" ")
    ngrams_total = 0
    for n in range(1, n_grams+1):
        for i in range(len(words) - n + 1):
            ngrams_total += 1
            ngram = " ".join([word.strip(".,!?/\\'\"-") for word in words[i:i+n]])
            if ngram in doc_freq:
                doc_freq[ngram] += 1
            else:
                doc_freq[ngram] = 1
    for word in doc_freq:
        doc_freq[word] = doc_freq[word] / ngrams_total
    return doc_freq

class Document:
    def __init__(self, text: str, metadata = None):
        self.id = str(uuid.uuid4())
        self.metadata = metadata
        self.text = text
    def __eq__(self, other: object) -> bool:
        return self.id == other.id
    def __hash__(self) -> int:
        return hash(self.id)
    def __str__(self) -> str:
        return json.dumps({"doc": self.metadata}, indent=4)
    def __repr__(self) -> str:
        return self.__str__()
    def pretty_print(self):
        print(json.dumps({"text": self.text}, indent=4))

class SearchItem:
    def __init__(self, document: Document, doc_freq: dict[str, float] = None):
        self.doc = document
        self.doc_freq = doc_freq
        self.df_idf = 0
    def set_df_idf(self, df_idf: float):
        self.df_idf = df_idf
    def __lt__(self, other):
        return self.df_idf < other.df_idf
    def __str__(self) -> str:
        return json.dumps({"df_idf": round(self.df_idf, 4), "id": self.doc.id, "doc": self.doc.text}, indent=4)
    def __repr__(self) -> str:
        return self.__str__()

class SearchIndex:
    def __init__(self):
        self.index: dict[str, list[SearchItem]] = {}
        self.word_freq: dict[str, int] = {}
        self.items: list[SearchItem] = []
        self.doc_count = 0
    
    def add_document(self, document: Document):
        doc_freq: dict[str, float] = get_doc_freq(document.text)
        self.doc_count += 1
        self.add_to_word_freq(doc_freq.keys())
        self.items.append(SearchItem(document, doc_freq))
        
    def build_index(self):
        print("Building index...")
        start = time()
        for item in tqdm.tqdm(self.items, "Indexing items"):
            for word in item.doc_freq.keys():
                df_idf = self.calc_df_idf(word, item.doc_freq[word])
                index_item = SearchItem(item.doc)
                index_item.set_df_idf(df_idf)
                if word in self.index:
                    heapq.heappush(self.index[word], index_item)
                else:
                    self.index[word] = [index_item]
        print(f"Index of {self.doc_count} items built in {time() - start} seconds")
        
    def calc_df_idf(self, word: str, df: float):
        normalized_df = df / self.word_freq[word]
        return normalized_df * math.log(self.doc_count / 1 + ( self.word_freq[word]))
    
    def add_to_word_freq(self, words: list[str]):
        for word in words:
            if word in self.word_freq:
                self.word_freq[word] += 1
            else:
                self.word_freq[word] = 1

    def search(self, **kwargs) -> list[SearchItem]:
        query = kwargs.get("query").lower()
        n = kwargs.get("n", 5)
        return heapq.nlargest(n, self.index.get(query, []))