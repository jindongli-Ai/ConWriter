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

## Perspective and Future Directions

ConWriter was proposed to explore a possible approach to consistency-aware long-form story generation: combining **static memory**, which maintains relatively stable narrative requirements and constraints, with **dynamic memory**, which tracks evolving story states. Lightweight **neuro-symbolic** checks are used to identify potential inconsistencies and guide local revisions.

Our experience suggests that external checking and repair cannot fully compensate for limited story-generation capability in the backbone LLM. If a model struggles with narrative planning, instruction following, causal coherence, or targeted rewriting, repeated validation may still fail to produce a satisfactory story. We therefore see **training or fine-tuning an LLM specifically for long-form storytelling** as a promising direction for future research. Improving the model's ability to plan and generate coherent narratives could make memory-based consistency control and neuro-symbolic verification more effective as complementary mechanisms. This is a research perspective, not an experimentally established result of ConWriter.

## Practical Considerations and Limitations

**Research scope.** ConWriter focuses on consistency-aware long-form story generation, as evaluated on ConStory-Bench. Its memory and neuro-symbolic components are intended to help detect and mitigate consistency violations; they do not guarantee error-free stories. The current framework is not an end-to-end system for multi-episode drama production, shot planning, or video generation.

**Backbone LLM dependence.** The pipeline requires the underlying LLM to follow scene-generation instructions, interpret narrative constraints, produce structured information, and perform targeted revisions. Models with weaker instruction-following or reasoning capabilities may encounter recurring validation failures or unsuccessful repairs. Model capability is an important practical consideration, but validation failures can also result from parsing or implementation mismatches and should not automatically be attributed to the model.

**Localized repair.** Sentence-level and other localized repairs depend on the LLM's ability to correct a violation while preserving surrounding prose. Repairs run within bounded attempts and may fail, leave violations unresolved, or introduce new problems. Additional validation does not by itself ensure a successful correction.

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

