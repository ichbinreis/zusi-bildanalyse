import re
import numpy as np

class SimpleTokenizer:
    def __init__(self):
        self.encoder = {
            "ein": 0, "test": 1, "bild": 2, "vergleich": 3, "zusi": 4, "modul": 5,
            "clip": 6, "objekt": 7, "ordner": 8, "ergebnis": 9, "analyse": 10
        }
        self.decoder = {v: k for k, v in self.encoder.items()}

    def tokenize(self, text):
        tokens = []
        for word in re.findall(r"\w+", text.lower()):
            tokens.append(self.encoder.get(word, 0))
        return np.array(tokens)
