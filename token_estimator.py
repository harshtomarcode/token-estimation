"""Count text tokens from local Qwen3-Embedding or BGE-M3 tokenizer.json files."""

import base64
import json
from pathlib import Path
import struct
import unicodedata

import regex


class _PrecompiledCharsMap:
    """Read BGE's embedded normalization table without SentencePiece/protobuf."""

    def __init__(self, encoded_map):
        blob = base64.b64decode(encoded_map, validate=True)
        trie_size = struct.unpack_from("<I", blob)[0]
        if not trie_size or trie_size % 4 or 4 + trie_size >= len(blob):
            raise ValueError("Invalid precompiled character map")
        self.trie = struct.unpack_from(f"<{trie_size // 4}I", blob, 4)
        self.replacements = blob[4 + trie_size:]

    def _lookup(self, text):
        # Darts bit layout: https://github.com/huggingface/spm_precompiled
        unit = self.trie[0]
        node = (unit >> 10) << ((unit & 512) >> 6)
        for byte in text.encode("utf-8"):
            if byte == 0:
                break
            node ^= byte
            unit = self.trie[node]
            if (unit & 0x800000FF) != byte:
                return None
            node ^= (unit >> 10) << ((unit & 512) >> 6)
            if unit & 256:
                start = self.trie[node] & 0x7FFFFFFF
                end = self.replacements.index(b"\0", start)
                return self.replacements[start:end].decode("utf-8")
        return None

    def normalize(self, text):
        # Match HF's grapheme/first-prefix behavior, not slow SentencePiece.
        # https://docs.rs/tokenizers/latest/src/tokenizers/normalizers/precompiled.rs.html
        result = []
        for match in regex.finditer(r"\X", text):
            grapheme = match.group()
            mapped = self._lookup(grapheme) if len(grapheme.encode("utf-8")) < 6 else None
            if mapped is not None:
                result.append(mapped)
            else:
                for character in grapheme:
                    mapped = self._lookup(character)
                    result.append(character if mapped is None else mapped)
        return "".join(result)


def _special_token_count(processor):
    """Read single-input BOS/EOS overhead from the supplied JSON revision."""
    if processor is None or processor["type"] == "ByteLevel":
        return 0
    if processor["type"] == "Sequence":
        return sum(_special_token_count(p) for p in processor["processors"])
    if processor["type"] != "TemplateProcessing":
        raise ValueError("Unsupported postprocessor; expected an embedding tokenizer.json")
    count = 0
    sequences = []
    for item in processor["single"]:
        if "SpecialToken" in item:
            name = item["SpecialToken"]["id"]
            count += len(processor["special_tokens"][name]["ids"])
        elif "Sequence" in item:
            sequences.append(item["Sequence"]["id"])
        else:
            raise ValueError("Unsupported single-input template")
    if sequences != ["A"]:
        raise ValueError("Expected exactly one input sequence in the template")
    return count


class TokenEstimator:
    """Load once, then call count(text). Only the stdlib and regex are used.

    Supports the published Qwen3-Embedding and BAAI/bge-m3 JSON pipelines.
    Counts are before truncation/padding. Service-side prompts must be included
    in text by the caller. This is not a general Hugging Face tokenizer loader.
    """

    def __init__(self, tokenizer_path: str | Path):
        path = Path(tokenizer_path)
        if path.is_dir():
            path /= "tokenizer.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data["model"]
        self.kind = model["type"]
        self.special_tokens = _special_token_count(data.get("post_processor"))

        # These models use literal added tokens, extracted before normalization.
        patterns = []
        for token in sorted(data.get("added_tokens", []), key=lambda t: -len(t["content"])):
            if token.get("normalized") or token.get("single_word") or not token["content"]:
                raise ValueError("Unsupported normalized/single-word/empty added token")
            patterns.append(
                (r"\s*" if token.get("lstrip") else "")
                + regex.escape(token["content"])
                + (r"\s*" if token.get("rstrip") else "")
            )
        self._added = regex.compile("|".join(patterns)) if patterns else None

        normalizer = data.get("normalizer")
        pre = data.get("pre_tokenizer") or {}
        if self.kind == "BPE":
            unsupported = (
                "dropout", "unk_token", "continuing_subword_prefix",
                "end_of_word_suffix", "fuse_unk", "byte_fallback", "ignore_merges",
            )
            if any(model.get(key) for key in unsupported):
                raise ValueError("Only Qwen's plain byte-level BPE configuration is supported")
            if normalizer not in (None, {"type": "NFC"}):
                raise ValueError("Expected Qwen's NFC normalizer (or no normalizer)")
            self._nfc = normalizer is not None
            stages = pre.get("pretokenizers", [])
            if pre.get("type") != "Sequence" or len(stages) != 2:
                raise ValueError("Expected Qwen's Split + ByteLevel pretokenizer")
            split, bytelevel = stages
            if (
                split.get("type") != "Split" or split.get("behavior") != "Isolated"
                or split.get("invert") or "Regex" not in split.get("pattern", {})
                or bytelevel.get("type") != "ByteLevel"
                or bytelevel.get("add_prefix_space", True)
                or bytelevel.get("use_regex", True)
            ):
                raise ValueError("Unsupported Qwen pretokenizer options")
            pattern = split["pattern"]["Regex"]
            if regex.compile(pattern).groups:
                raise ValueError("The splitting pattern must not contain capturing groups")
            self._split = regex.compile("(" + pattern + ")")
            # Reversible GPT-2 byte alphabet, also used by Qwen.
            visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
            missing = [b for b in range(256) if b not in visible]
            characters = visible + list(range(256, 256 + len(missing)))
            self._byte_chars = dict(zip(visible + missing, map(chr, characters)))
            if not all(c in model["vocab"] for c in self._byte_chars.values()):
                raise ValueError("BPE vocabulary does not contain the full byte alphabet")
            self._ranks = {
                tuple(pair.split(" ") if isinstance(pair, str) else pair): rank
                for rank, pair in enumerate(model["merges"])
            }
        elif self.kind == "Unigram":
            if model.get("byte_fallback") or model.get("unk_id") is None:
                raise ValueError("Expected BGE-M3 Unigram without byte fallback")
            norms = (normalizer or {}).get("normalizers", [])
            if (
                (normalizer or {}).get("type") != "Sequence" or len(norms) != 2
                or norms[0].get("type") != "Precompiled"
                or norms[1] != {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}
            ):
                raise ValueError("Expected BGE-M3's Precompiled + space-collapse normalizer")
            if (
                pre.get("type") != "Metaspace" or pre.get("replacement") != "▁"
                or not pre.get("add_prefix_space", True)
                or pre.get("prepend_scheme", "always") != "always"
                or not pre.get("split", True)
            ):
                raise ValueError("Expected BGE-M3's prefix-space Metaspace pretokenizer")
            self._normalizer = _PrecompiledCharsMap(norms[0]["precompiled_charsmap"])
            self._vocab = dict(model["vocab"])
            self._max_piece = max(map(len, self._vocab))
            self._unk_piece = model["vocab"][model["unk_id"]][0]
            self._unk_score = min(self._vocab.values()) - 10.0
        else:
            raise ValueError("Only Qwen3-Embedding BPE and BGE-M3 Unigram are supported")

    def count(self, text: str, *, add_special_tokens: bool = True) -> int:
        """Count one text, including JSON-defined BOS/EOS tokens by default.

        Literal added tokens in text count as one even when add_special_tokens
        is False; that flag controls only automatically inserted tokens.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        total = self.special_tokens if add_special_tokens else 0
        start = 0
        if self._added is not None:
            for match in self._added.finditer(text):
                total += self._count_text(text[start:match.start()]) + 1
                start = match.end()
        return total + self._count_text(text[start:])

    def _count_text(self, text):
        if not text:
            return 0
        if self.kind == "BPE":
            if self._nfc:
                text = unicodedata.normalize("NFC", text)
            # Isolated Split keeps both the matches and any unmatched spans.
            return sum(self._count_bpe(piece) for piece in self._split.split(text) if piece)
        text = regex.sub(" {2,}", " ", self._normalizer.normalize(text))
        if not text:
            return 0
        text = text.replace(" ", "▁")
        if not text.startswith("▁"):
            text = "▁" + text
        # Each marker starts the next piece; a trailing marker is its own piece.
        return sum(self._count_unigram(m.group()) for m in regex.finditer(r"▁[^▁]*", text))

    def _count_bpe(self, text):
        pieces = [self._byte_chars[b] for b in text.encode("utf-8")]
        while len(pieces) > 1:
            rank, index = min(
                (self._ranks.get((pieces[i], pieces[i + 1]), float("inf")), i)
                for i in range(len(pieces) - 1)
            )
            if rank == float("inf"):
                break
            # Lowest rank first, leftmost pair on ties; vocabulary IDs are not ranks.
            pieces[index:index + 2] = [pieces[index] + pieces[index + 1]]
        return len(pieces)

    def _count_unigram(self, text):
        # Viterbi: maximize total vocabulary score, not longest match or fewest tokens.
        # https://docs.rs/tokenizers/latest/src/tokenizers/models/unigram/model.rs.html
        scores = [0.0] + [float("-inf")] * len(text)
        previous = [0] * (len(text) + 1)
        unknown = [False] * (len(text) + 1)
        for start in range(len(text)):
            candidates = []
            for end in range(start + 1, min(len(text), start + self._max_piece) + 1):
                piece = text[start:end]
                if piece in self._vocab:
                    candidates.append((end, self._vocab[piece], piece == self._unk_piece))
            if text[start] not in self._vocab:
                candidates.append((start + 1, self._unk_score, True))
            for end, score, is_unknown in candidates:
                candidate = scores[start] + score
                if candidate > scores[end]:
                    scores[end] = candidate
                    previous[end] = start
                    unknown[end] = is_unknown
        count, end, last_unknown = 0, len(text), False
        while end:
            count += not (unknown[end] and last_unknown)
            last_unknown = unknown[end]
            end = previous[end]
        return count
