# ConWriter

Official implementation of **ConWriter: Transition-Constrained Stateful Long-Form Story Generation with Lightweight Neuro-Symbolic Consistency Control** (EMNLP Findings 2026).

## Contents

- `ConWriter/`: framework source code
- `scripts/run_conwriter.py`: minimal runnable example
- `ConWriter/config/default.yaml`: default configuration

## Install

```bash
pip install -r requirements.txt
```

## Run a local example

```bash
python scripts/run_conwriter.py "Write a concise story about Alice and Bob."
```

LLM-backed generation is optional. Set credentials through environment variables and enable it in the YAML configuration. Never commit credentials.

The implementation includes deterministic fallbacks for environments without an API provider. Benchmark data and evaluation tooling are maintained separately and are not included in this repository.

## Citation

Please cite the accompanying EMNLP Findings 2026 paper.

```
@article{li2026conwriter,
  title={ConWriter: Transition-Constrained Stateful Long-Form Story Generation with Lightweight Neuro-Symbolic Consistency Control},
  author={Li, Jindong and Yang, Yang and Liu, Zihao and Yue, Yutao and Yang, Menglin},
  journal={arXiv preprint arXiv:2608.05169},
  year={2026}
}
```

