# Token estimation with local tokenizer files

Count text tokens for **Qwen3-Embedding** and **BAAI/bge-m3**, using Python 3.10+
and **`regex` as the only dependency**. Everything runs locally, with no model
weights, network calls, SentencePiece, Transformers, Tokenizers, or PyTorch.
The implementation is a single file: [token_estimator.py](token_estimator.py).

```sh
python -m pip install -r requirements.txt
```

Supply **`tokenizer.json` from the exact model revision your embedding service
uses**. It contains the vocabulary, merge rules or vocabulary scores,
normalization, added tokens, and automatic special-token template. You do not
need `tokenizer_config.json`, `vocab.json`, `merges.txt`, or `.model` files for
this implementation. A binary SentencePiece `.model` alone is not supported.

```python
from token_estimator import TokenEstimator

# Pass a directory containing tokenizer.json, or the JSON file itself.
qwen = TokenEstimator("/path/to/Qwen3-Embedding/tokenizer.json")
bge = TokenEstimator("/path/to/bge-m3/tokenizer.json")

text = "Hello, world!"
print(qwen.count(text))
print(bge.count(text))

# Count only the supplied text, without automatically inserted BOS/EOS tokens.
print(qwen.count(text, add_special_tokens=False))
```

Load each estimator once and reuse it. The JSON files are about 11.4 MB for
Qwen3-Embedding-0.6B and 17.1 MB for BGE-M3; Python dictionaries use more memory
than the files on disk. Dependencies are small, but the vocabulary must still
be loaded.

Qwen counting applies the JSON's NFC normalization and Unicode splitting
pattern, maps UTF-8 bytes to its byte alphabet, then performs ranked BPE merges.
BGE counting reads its embedded normalization table directly, applies its
space rules, and finds the highest-scoring Unigram segmentation. It also fuses
consecutive unknown characters into one unknown token. Neither path uses a
characters-per-token heuristic.

`count()` includes automatic special tokens by default, reading their count
from the JSON instead of hardcoding it. The inspected files add one EOS token
for Qwen and two boundary tokens for BGE. Literal added-token strings such as
`<mask>` or `<|endoftext|>` also count as tokens; disabling automatic tokens
does not disable recognition of those strings.

This deliberately supports these two tokenizer pipelines, not arbitrary
Hugging Face models. Unsupported pipeline options raise `ValueError`. Counts
are **before truncation and padding**. Include any query instruction or prefix
in the text you pass; no chat template or service-side prompt is added here.

The target is the supplied **fast `tokenizer.json` pipeline**. An embedding
service using the slow SentencePiece tokenizer, config overrides, different
Unicode tables, or additional preprocessing can differ. Treat results as
estimates until compared with reference counts from your actual service;
there is no guaranteed error margin or upper bound. The simple Python BPE
loop is quadratic for a single very long piece, so it is suited to ordinary
text rather than huge unbroken strings.

Local smoke checks used the public Qwen3-Embedding-0.6B and BGE-M3 files,
covering multilingual text, accents, emoji, whitespace, and added tokens.
The official Unigram example and hand-traced merge/unknown-token cases passed.
For `"Hello, world!"`, this implementation returns **5** for Qwen and **6** for
BGE, including automatic tokens. The native GGUF benchmark below measures deployment-specific differences
against an independent tokenizer implementation.

Sources: [Qwen tokenizer JSON](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/tokenizer.json),
[BGE-M3 tokenizer JSON](https://huggingface.co/BAAI/bge-m3/blob/main/tokenizer.json),
[Hugging Face Unigram](https://docs.rs/tokenizers/latest/src/tokenizers/models/unigram/model.rs.html),
[BGE normalization-map format](https://github.com/huggingface/spm_precompiled).

## Evaluation results

Evaluated on September 23, 2026 against the locally deployed **BGE-M3 Q8_0**
and **Qwen3-Embedding-0.6B Q8_0** GGUF tokenizers. The estimator was evaluated
unchanged, without calibration on the corpus.

The corpus contains **230 sources**: 100 Wikipedia pages, 100 code files,
10 company annual reports, and 20 golden Excel workbooks. Each complete
source was counted, plus 512 deterministic excerpts: 1,024 characters from
the beginning, 4,096 from the middle, and 16,384 from the end where the source
was longer. Another 30 synthetic inputs probe Unicode, whitespace, empty
strings, and literal special tokens. All **1,544 comparisons** succeeded.

Across the **742 document/excerpt inputs**, Qwen matched **742/742** counts
exactly. BGE matched **294/742** and overcounted the other **448 by one token**.

| Model | Input group | Exact counts | Mean absolute error (tokens) | Mean absolute % error | Maximum absolute error (tokens) |
|---|---|---:|---:|---:|---:|
| BGE-M3 | Complete documents | 0/230 | 1.000 | 0.0693% | 1 |
| BGE-M3 | Excerpts | 294/512 | 0.426 | 0.0649% | 1 |
| BGE-M3 | Synthetic diagnostics | 21/30 | 0.400 | 9.2514% | 3 |
| Qwen3-Embedding-0.6B | Complete documents | 230/230 | 0.000 | 0.0000% | 0 |
| Qwen3-Embedding-0.6B | Excerpts | 512/512 | 0.000 | 0.0000% | 0 |
| Qwen3-Embedding-0.6B | Synthetic diagnostics | 27/30 | 0.333 | 3.4392% | 4 |

Mean absolute percentage error averages `100 * abs(estimate - truth) / truth`
per input. Synthetic diagnostics are kept separate because they intentionally
stress uncommon cases, often with very short inputs.

Every BGE corpus mismatch coincided with trailing whitespace: the supplied
Hugging Face JSON pipeline retains a final space marker that the GGUF runtime
removes. All complete documents ended with whitespace, explaining the zero
exact matches despite an error of only one token each. The largest diagnostic
errors involved combining-mark normalization and literal special-token
handling, including BGE's `<mask>` and Qwen's decomposed accents.

Ground truth used the GGUF vocabularies through LM Studio's installed native
llama.cpp library, with model weights skipped and truncation disabled.
Both counters included their automatic special tokens. Native token IDs
matched the running service on **42/42 short probes**. The HTTP embedding
usage fields returned zero, and the live tokenization RPC truncated long
inputs at 8,192 tokens, so full-document references came from the untruncated
native tokenizer.

Both counters received identical text: plain Wikipedia text, code, PDF text
extracted with `pdftotext -layout`, or serialized Excel cells and formulas
with available cached values. The sample was selected for coverage, not
randomly; excerpts share their source documents. These measurements apply
to the inspected tokenizer artifacts and runtime, and are **not a guaranteed
error bound** for arbitrary text or other deployments.

This repository contains the estimator, dependency list, and this usage and
results summary. Evaluation datasets, tokenizer/model assets, raw results,
and collection/evaluation scripts are kept locally and excluded from Git.
