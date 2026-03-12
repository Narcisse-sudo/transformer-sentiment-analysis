import os
import re
from collections import Counter

import torch


def clean_text(text: str) -> str:
    """Nettoyage simple des textes."""
    text = text.lower()
    text = re.sub(r"[^a-z'àâäéèêëîïôöùûüç0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_vocab(texts, vocab_size: int):
    """Cree un vocabulaire a partir des textes."""
    counter = Counter()

    for text in texts:
        cleaned = clean_text(text)
        tokens = cleaned.split()
        counter.update(tokens)

    # Prendre les mots les plus frequents
    most_common = counter.most_common(vocab_size - 2)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, _) in enumerate(most_common):
        vocab[word] = i + 2

    return vocab


def load_and_prepare_imdb(path, vocab=None, vocab_size: int = 20000, max_len: int = 256):
    """Charge les donnees IMDb et prepare des tenseurs pour le modele."""
    path = str(path)
    texts = []
    labels = []

    # Charger les textes bruts
    for label, folder in [(1, "pos"), (0, "neg")]:
        folder_path = os.path.join(path, folder)

        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                texts.append(text)
                labels.append(label)

    # Creer le vocabulaire
    if vocab is None:
        vocab = create_vocab(texts, vocab_size)

    # Convertir textes en indices
    encoded_texts = []
    for text in texts:
        text = clean_text(text)
        tokens = text.split()[:max_len]
        indices = [vocab.get(token, 1) for token in tokens]  # 1 = <UNK>
        indices += [vocab["<PAD>"]] * (max_len - len(indices))  # 0 = <PAD>
        encoded_texts.append(indices)

    return torch.tensor(encoded_texts), torch.tensor(labels), vocab
