# Prompt Optimization Results: GEPA vs LangSmith

## Shopping Concierge (prompt-opt-demo)

### LLM Calls
| | LangSmith | GEPA |
|--|--|--|
| Agent runs | ~71 | 1,708 |
| Wall time | ~20 min | 201 min |
| Relative cost | 1× | ~24× |

### Results
| Metric | Baseline | LangSmith (train) | LangSmith (val) | GEPA (train) | GEPA (val) |
|--|--|--|--|--|--|
| Task completeness | 49% | 60% | 62% | 69% | 65% |
| Critical agents called | 58% | 94% | 93% | 97% | 89% |
| Sequence respected | 59% | 91% | 91% | 94% | 86% |
| Composite | 0.583 | — | ~0.820 | 0.842 | 0.745 |

LangSmith train/val gap: ~1% (generalized well)
GEPA train/val gap: ~10% (overfitting signature)

### Key files
- Baseline prompt: prompts/baseline.md
- LangSmith optimized: prompts/optimized.md (38 lines)
- GEPA best (iter 20): gepa/composite/prompt_2026-03-25T12-59-07.txt (270 lines)
- GEPA stats: gepa/composite/stats_2026-03-25T12-59-07.json
  - baseline_train: 0.630, baseline_val: 0.583
  - best_train: 0.842 (iter 20), optimized_val: 0.745
  - total_agent_calls: 1708, duration: 201.6 min
