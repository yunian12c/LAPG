# Chinese CAPT GNN (LAPG/CN)

SSL base → Prosody FiLM → Qwen FiLM. Graph: Initial → Final → Tone → Global.

## Layout

```
LAPG/CN/
  train_gop_graph.py          # training entry (paths relative to this dir)
  phoneme_gnn.py              # model
  *.py                        # graph / FiLM / APA / cache helpers
  data/{train,test}/labels.jsonl
  features/{hubert_feature,tencent_wav2vec2_feature,wavlm_feature,qwen_feature_14layer}/
  caches/{fa_hubert_native,hubert_f0,hubert_energy}/
  run_seeds.sh                # 5 best seeds
```

## Best 5 seeds (by `apa_mean_pcc`)

| Seed | apa_mean_pcc (ref) |
|------|--------------------|
| 180  | 0.7968 |
| 265  | 0.7965 |
| 70   | 0.7962 |
| 226  | 0.7953 |
| 12   | 0.7951 |

## Run

```bash
cd CN          # under LAPG/
CUDA_DEVICE=0 ./run_seeds.sh
CUDA_DEVICE=0 ./run_seeds.sh 180 265
```

Defaults in `train_gop_graph.py` resolve from the script directory (`data/`, `features/`, `caches/`, `exp/`).  
Python: set `PY=python` if needed, or use conda env `GNN`.
