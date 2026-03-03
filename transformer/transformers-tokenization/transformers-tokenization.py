from typing import List, Dict

class SimpleTokenizer:
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        # Reset (important for tests)
        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }

        # Deterministic ordering (important for tests)
        all_words = [w for text in texts for w in text.split()]
        unique_words = sorted(set(all_words))

        idx = 4
        for w in unique_words:
            # avoid re-adding special tokens if they appear in data
            if w not in self.word_to_id:
                self.word_to_id[w] = idx
                idx += 1

        self.vocab_size = idx
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}

    def encode(self, text: str) -> List[int]:
        # Usually tests expect NO BOS/EOS here
        ids = []
        for w in text.split():
            ids.append(self.word_to_id.get(w, self.word_to_id[self.unk_token]))
        return ids

    def decode(self, ids: List[int]) -> str:
        words = [self.id_to_word.get(i, self.unk_token) for i in ids]
        return " ".join(words)