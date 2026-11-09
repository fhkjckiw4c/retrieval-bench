# retrieval-bench

A small, file-in/file-out harness for scoring ranked retrieval runs against
graded qrels.

Most reported retrieval numbers cannot be recomputed from the paper text
alone: cutoffs, tie-breaking, and duplicate handling differ between scripts.
Here a run file plus a qrels file is enough to reproduce a score exactly,
and both are plain JSONL. The harness is deliberately model-agnostic; it
scores whatever ranking an external BM25 or dense encoder produced.

## Install

Python 3.10 or newer, no runtime dependencies:

```sh
python -m pip install -e .
```

## Usage

```sh
python -m retrieval_bench.evaluate \
    --qrels data/toy_qrels.jsonl \
    --run data/toy_run.bm25.jsonl \
    --metrics recall@10,mrr@10,ndcg@10 \
    --label bm25
```

The command prints a JSON summary and exits non-zero only on malformed
input:

```json
{
  "label": "bm25",
  "queries": 3,
  "missing": 0,
  "metrics": {
    "mrr@10": 0.8333,
    "ndcg@10": 0.7863,
    "recall@10": 0.8889
  }
}
```

Add `--per-query out.json` to dump the individual scores, which is what
you want for error analysis or for feeding a significance test.

Queries that are in the qrels but absent from the run are counted in
`missing` and left out of the mean, so a truncated export shows up instead
of dragging every score toward zero.

### File formats

```jsonc
// qrels: positive relevance is graded, 0 means judged-not-relevant
{"query_id": "q1", "doc_id": "d1", "relevance": 2}

// run: one row per scored document; order in the file does not matter
{"query_id": "q1", "doc_id": "d1", "score": 11.2}
```

Runs are sorted by descending score with doc id as the tie-break. Within a
query a repeated doc id keeps its first occurrence.

## Metrics

| metric     | definition                                                    |
|------------|---------------------------------------------------------------|
| `recall@k` | share of relevant docs found in the top k                     |
| `mrr@k`    | reciprocal rank of the first relevant doc, 0 if none in top k |
| `ndcg@k`   | graded nDCG using the qrels relevance values as gains         |

All metrics are macro-averaged over queries and are computed at the cutoff
you pass; there is no built-in default that changes underneath you.

## Layout

```
retrieval_bench/metrics.py    cutoff-aware metric functions
retrieval_bench/dataset.py    validated JSONL qrels/run loaders
retrieval_bench/evaluate.py   CLI and macro averaging
data/toy_qrels.jsonl          3-query example qrels
data/toy_run.*.jsonl          two example runs that disagree
```

## Results

Scores on the bundled three-query example, produced with `--metrics
recall@10,mrr@10,ndcg@10`:

| system | recall@10 | mrr@10 | ndcg@10 |
|--------|-----------|--------|---------|
| bm25   | 0.889     | 0.833  | 0.786   |
| dense  | 1.000     | 0.833  | 0.865   |

Same documents, different order: the metric matters because recall only
counts what is retrieved at all while nDCG also rewards the graded hit in
the first slot. Replace this table with your own runs; the toy set is too
small to compare systems on.

## Limitations

- Unjudged documents are treated as non-relevant. Pooling is out of scope.
- Macro average over queries only; no per-collection or per-slice weighting.
- No significance testing yet; bootstrap intervals are in review.
- Ties are broken by doc id, which is deterministic but arbitrary.
- The toy data is illustrative, not a benchmark.

## Prior work

- Järvelin and Kekäläinen, *Cumulated gain-based evaluation of IR
  techniques*, TOIS 2002.
- Thakur et al., *BEIR: A heterogeneous benchmark for zero-shot evaluation
  of information retrieval models*, NeurIPS 2021.
- Muennighoff et al., *MTEB: Massive text embedding benchmark*, 2022.
- `trec_eval` for the reference semantics of the metrics above.

## License

MIT, see [LICENSE](LICENSE).
