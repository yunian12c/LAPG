"""HMamba-style classic MDD eval: dump hyp/ref/human and run mdd_result.sh."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any


def ids_to_phone_str(ids: list[int], id2tok: dict[int, str], skip: set[int] | None = None) -> str:
    skip = skip or set()
    toks = []
    for i in ids:
        if i in skip:
            continue
        t = id2tok.get(int(i), "<unk>")
        if t in ("<blank>", "<pad>", "<eps>", "<unk>"):
            if t == "<unk>":
                toks.append(t)
            continue
        # Safety: strip stress if an old 117-phone checkpoint dumps AA0/AA1.
        if len(t) >= 2 and t[-1] in "012" and t[-2].isalpha():
            t = t[:-1]
        toks.append(t)
    return " ".join(toks)


def write_kaldi_text(path: Path, utt2seq: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for utt in sorted(utt2seq.keys()):
            seq = utt2seq[utt].strip()
            # align-text needs at least one token; use <unk> placeholder if empty
            if not seq:
                seq = "<unk>"
            f.write(f"{utt} {seq}\n")


def parse_classic_mdd_output(text: str) -> dict[str, float]:
    """Parse ins_del_sub_cor_analysis.py + compute-wer stdout."""
    out: dict[str, float] = {}
    m = re.search(r"Precision:\s*([0-9.]+)", text)
    if m:
        out["mdd_precision"] = float(m.group(1))
    m = re.search(r"Recall:\s*([0-9.]+)", text)
    if m:
        out["mdd_recall"] = float(m.group(1))
    m = re.search(r"F1:\s*([0-9.]+)", text)
    if m:
        out["mdd_f1"] = float(m.group(1))
    m = re.search(r"%WER\s+([0-9.]+)", text)
    if m:
        out["per"] = float(m.group(1)) / 100.0
    m = re.search(r"Correct Diag:\s*([0-9.]+)", text)
    if m:
        out["mdd_correct_diag"] = float(m.group(1))
    m = re.search(r"DER:\s*([0-9.]+)", text)
    if m:
        out["mdd_der"] = float(m.group(1))
    return out


def run_classic_mdd(
    *,
    work_dir: Path,
    hyp: dict[str, str],
    ref: dict[str, str],
    human: dict[str, str],
    eval_mdd_root: Path | None = None,
) -> dict[str, Any]:
    """Write sequences and run HMamba mdd_result.sh. Returns metrics + raw log path."""
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    # Use *.in so mdd_result.sh can safely write filtered hyp/ref/human_seq in the same dir.
    hyp_p = work_dir / "hyp.in"
    ref_p = work_dir / "ref.in"
    human_p = work_dir / "human.in"
    write_kaldi_text(hyp_p, hyp)
    write_kaldi_text(ref_p, ref)
    write_kaldi_text(human_p, human)

    if eval_mdd_root is None:
        eval_mdd_root = Path(__file__).resolve().parent / "eval_mdd"
    else:
        eval_mdd_root = Path(eval_mdd_root).resolve()
    script = eval_mdd_root / "mdd_result.sh"
    if not script.is_file():
        raise FileNotFoundError(f"missing {script}")

    log_path = work_dir / "mdd_result.log"
    # Absolute paths required: mdd_result.sh cds into work_dir.
    cmd = [
        "bash",
        str(script),
        str(human_p.resolve()),
        str(ref_p.resolve()),
        str(hyp_p.resolve()),
        str(work_dir),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PATH": os.environ.get("PATH", ""),
        },
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.write_text(combined, encoding="utf-8")
    metrics = parse_classic_mdd_output(combined)
    metrics["mdd_eval_ok"] = 1.0 if ("mdd_f1" in metrics and "per" in metrics) else 0.0
    metrics["mdd_eval_rc"] = float(proc.returncode)
    return metrics
