# English CAPT GNN (LAPG/EN)

Qwen → GOP FiLM → SSL FiLM (`film_order=qwen_gop_ssl`). Graph: Phone ↔ Word.

## Layout

```
LAPG/EN/
  train_speechocean2.py       # training entry (paths relative to this dir)
  english_gnn2.py             # English Phone–Word GNN model
  *.py                        # helpers (graph / FiLM / APA / GOP / MDD)
  data/{train,test}/labels.jsonl + utt_ids.txt
  features/{hubert,tencent_wav2vec2,wavlm,qwen_feature_14layer}/
  gop_feature/{raw_kaldi_gop/librispeech,seq_data_librispeech}/
  caches/{fa_cache,hubert_f0,hubert_energy}/
  resource/vocab_merge.json
  eval_mdd/                   # classic MDD scoring helpers
  run_seeds.sh                # 5 main-table seeds
```

## Main-table 5 seeds

| Seed | mdd_f1 (ref) |
|------|--------------|
| 173  | ~0.650 |
| 185  | ~0.655 |
| 79   | ~0.646 |
| 237  | ~0.663 |
| 239  | ~0.658 |

## Run

```bash
cd EN          # under LAPG/
CUDA_DEVICE=0 ./run_seeds.sh
CUDA_DEVICE=0 ./run_seeds.sh 173 185
```

Defaults resolve from this directory (`data/`, `features/`, `gop_feature/`, `caches/`, `resource/`, `exp/`).  
Python: set `PY=python` if needed, or use conda env `GNN`.
