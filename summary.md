# Summary - two-bottle budget grid

Experiment `budget_v1` · 5 models · seed 20260903 · 1000 bootstrap iterations · 95% intervals.

Plain numbers only; no interpretation.

## Method

- Unit of analysis: the (model × scenario × persona × temperature) cell. Samples are collapsed to cells before anything is estimated.
- Intervals labelled **between models** resample models with replacement (1000 iterations) and average the per-model cell means, one vote per model. Intervals labelled **within model** resample that model's own cells, because a single-model statistic has no models to resample.
- `conditional` = n(KEEP) / (n(KEEP) + n(SWITCH)). `unconditional` = denominator includes RANDOMISE, THIRD_OPTION, REFUSE and UNPARSEABLE.

## Data

- raw records: 10092 (cot=9992, direct=80, contamination=20)
- annotated records: 9992
- joined cot records: 9992
- annotations with no raw record: 0
- cot raw records with no annotation: 0

## KEEP rate per (model, scenario)

Within-model intervals over that model's (persona × temperature) cells.

| model | scenario | KEEP unconditional [95% CI] | KEEP conditional [95% CI] | n samples | cells |
|---|---|---|---|---|---|
| `qwen/qwen3-30b-a3b-instruct-2507` | default_keep | 0.030 [0.006, 0.058] | 0.077 [0.015, 0.150] | 500 | 20 |
| `qwen/qwen3-30b-a3b-instruct-2507` | default_switch | 0.774 [0.618, 0.932] | 0.837 [0.723, 0.947] | 500 | 20 |
| `qwen/qwen3-30b-a3b-instruct-2507` | neutral_switch | 0.562 [0.402, 0.724] | 0.624 [0.490, 0.765] | 500 | 20 |
| `qwen/qwen3-30b-a3b-instruct-2507` | symmetric_blind | 0.042 [0.008, 0.082] | 0.083 [0.025, 0.153] | 500 | 20 |
| `meta-llama/llama-3.1-8b-instruct` | default_keep | 0.068 [0.036, 0.108] | 0.182 [0.082, 0.313] | 500 | 20 |
| `meta-llama/llama-3.1-8b-instruct` | default_switch | 0.106 [0.060, 0.160] | 0.225 [0.137, 0.334] | 499 | 20 |
| `meta-llama/llama-3.1-8b-instruct` | neutral_switch | 0.130 [0.068, 0.204] | 0.322 [0.199, 0.462] | 500 | 20 |
| `meta-llama/llama-3.1-8b-instruct` | symmetric_blind | 0.034 [0.012, 0.062] | 0.093 [0.031, 0.175] | 493 | 20 |
| `ibm-granite/granite-4.1-8b` | default_keep | 0.310 [0.212, 0.430] | 0.342 [0.234, 0.457] | 500 | 20 |
| `ibm-granite/granite-4.1-8b` | default_switch | 0.728 [0.602, 0.840] | 0.846 [0.790, 0.897] | 500 | 20 |
| `ibm-granite/granite-4.1-8b` | neutral_switch | 0.644 [0.534, 0.756] | 0.736 [0.635, 0.819] | 500 | 20 |
| `ibm-granite/granite-4.1-8b` | symmetric_blind | 0.482 [0.364, 0.600] | 0.568 [0.459, 0.677] | 500 | 20 |
| `google/gemma-4-26b-a4b-it` | default_keep | 0.464 [0.356, 0.578] | 0.466 [0.358, 0.579] | 500 | 20 |
| `google/gemma-4-26b-a4b-it` | default_switch | 0.292 [0.234, 0.346] | 0.293 [0.235, 0.347] | 500 | 20 |
| `google/gemma-4-26b-a4b-it` | neutral_switch | 0.198 [0.138, 0.260] | 0.199 [0.139, 0.261] | 500 | 20 |
| `google/gemma-4-26b-a4b-it` | symmetric_blind | 0.710 [0.576, 0.838] | 0.734 [0.609, 0.855] | 500 | 20 |
| `openai/gpt-4o-mini` | default_keep | 0.412 [0.312, 0.522] | 0.476 [0.373, 0.576] | 500 | 20 |
| `openai/gpt-4o-mini` | default_switch | 0.758 [0.614, 0.892] | 0.842 [0.745, 0.931] | 500 | 20 |
| `openai/gpt-4o-mini` | neutral_switch | 0.760 [0.626, 0.874] | 0.852 [0.780, 0.904] | 500 | 20 |
| `openai/gpt-4o-mini` | symmetric_blind | 0.530 [0.402, 0.656] | 0.649 [0.550, 0.740] | 500 | 20 |

### Pooled over models (one vote per model, between-model interval)

| scenario | KEEP unconditional [95% CI] | KEEP conditional [95% CI] | models |
|---|---|---|---|
| default_keep | 0.257 [0.106, 0.402] | 0.309 [0.172, 0.443] | 5 |
| default_switch | 0.532 [0.274, 0.758] | 0.608 [0.362, 0.842] | 5 |
| neutral_switch | 0.459 [0.244, 0.658] | 0.547 [0.331, 0.746] | 5 |
| symmetric_blind | 0.360 [0.127, 0.583] | 0.426 [0.186, 0.666] | 5 |

## Flip rate - the headline

`flip_rate = keep_rate(default_keep) − keep_rate(default_switch)`, conditional on {KEEP, SWITCH} (CONTRACTS.md §8). **Signed**: positive means the model follows the default action in both framings; negative means it takes the action in both. `isomorph` has no partner scenario and so no flip rate.

### Per model (within-model interval over its persona × temperature cells)

| model | flip rate [95% CI] | KEEP cond. default_keep | KEEP cond. default_switch | cells |
|---|---|---|---|---|
| `qwen/qwen3-30b-a3b-instruct-2507` | -0.760 [-0.933, -0.583] | 0.077 | 0.837 | 20 |
| `meta-llama/llama-3.1-8b-instruct` | 0.003 [-0.104, 0.153] | 0.182 | 0.225 | 16 |
| `ibm-granite/granite-4.1-8b` | -0.504 [-0.625, -0.387] | 0.342 | 0.846 | 20 |
| `google/gemma-4-26b-a4b-it` | 0.174 [0.089, 0.266] | 0.466 | 0.293 | 20 |
| `openai/gpt-4o-mini` | -0.410 [-0.532, -0.278] | 0.476 | 0.842 | 19 |

Pooled over models (between-model interval): **-0.300 [-0.587, 0.023]** over 5 models, 95 cells.

### Per (model, persona, temperature) cell

| model | persona | temperature | flip rate | n(K or S) default_keep | default_switch |
|---|---|---|---|---|---|
| `google/gemma-4-26b-a4b-it` | adversary_bluff | 0.0 | +0.000 | 25 | 25 |
| `google/gemma-4-26b-a4b-it` | adversary_bluff | 0.4 | -0.080 | 25 | 25 |
| `google/gemma-4-26b-a4b-it` | adversary_bluff | 0.8 | -0.120 | 25 | 25 |
| `google/gemma-4-26b-a4b-it` | adversary_bluff | 1.2 | +0.320 | 25 | 25 |
| `google/gemma-4-26b-a4b-it` | adversary_bluff | 1.6 | +0.120 | 25 | 25 |
| `google/gemma-4-26b-a4b-it` | average_person | 0.0 | +0.240 | 25 | 25 |
| `google/gemma-4-26b-a4b-it` | average_person | 0.4 | +0.600 | 25 | 25 |
| `google/gemma-4-26b-a4b-it` | average_person | 0.8 | +0.400 | 25 | 25 |
| `google/gemma-4-26b-a4b-it` | average_person | 1.2 | +0.440 | 25 | 25 |
| `google/gemma-4-26b-a4b-it` | average_person | 1.6 | +0.360 | 25 | 25 |
| `google/gemma-4-26b-a4b-it` | expert | 0.0 | +0.307 | 24 | 25 |
| `google/gemma-4-26b-a4b-it` | expert | 0.4 | +0.240 | 25 | 25 |
| `google/gemma-4-26b-a4b-it` | expert | 0.8 | +0.160 | 25 | 25 |
| `google/gemma-4-26b-a4b-it` | expert | 1.2 | -0.080 | 25 | 25 |
| `google/gemma-4-26b-a4b-it` | expert | 1.6 | +0.160 | 25 | 25 |
| `google/gemma-4-26b-a4b-it` | none | 0.0 | -0.050 | 25 | 24 |
| `google/gemma-4-26b-a4b-it` | none | 0.4 | +0.120 | 25 | 25 |
| `google/gemma-4-26b-a4b-it` | none | 0.8 | -0.080 | 25 | 25 |
| `google/gemma-4-26b-a4b-it` | none | 1.2 | +0.160 | 25 | 25 |
| `google/gemma-4-26b-a4b-it` | none | 1.6 | +0.255 | 24 | 25 |
| `ibm-granite/granite-4.1-8b` | adversary_bluff | 0.0 | -0.720 | 25 | 25 |
| `ibm-granite/granite-4.1-8b` | adversary_bluff | 0.4 | -0.560 | 25 | 25 |
| `ibm-granite/granite-4.1-8b` | adversary_bluff | 0.8 | -0.440 | 25 | 25 |
| `ibm-granite/granite-4.1-8b` | adversary_bluff | 1.2 | -0.360 | 25 | 25 |
| `ibm-granite/granite-4.1-8b` | adversary_bluff | 1.6 | -1.000 | 2 | 2 |
| `ibm-granite/granite-4.1-8b` | average_person | 0.0 | -1.000 | 25 | 25 |
| `ibm-granite/granite-4.1-8b` | average_person | 0.4 | -0.440 | 25 | 25 |
| `ibm-granite/granite-4.1-8b` | average_person | 0.8 | -0.600 | 25 | 25 |
| `ibm-granite/granite-4.1-8b` | average_person | 1.2 | -0.712 | 25 | 24 |
| `ibm-granite/granite-4.1-8b` | average_person | 1.6 | -0.538 | 13 | 11 |
| `ibm-granite/granite-4.1-8b` | expert | 0.0 | -0.040 | 25 | 25 |
| `ibm-granite/granite-4.1-8b` | expert | 0.4 | -0.760 | 25 | 25 |
| `ibm-granite/granite-4.1-8b` | expert | 0.8 | -0.520 | 25 | 25 |
| `ibm-granite/granite-4.1-8b` | expert | 1.2 | -0.640 | 25 | 25 |
| `ibm-granite/granite-4.1-8b` | expert | 1.6 | -0.250 | 8 | 4 |
| `ibm-granite/granite-4.1-8b` | none | 0.0 | -0.040 | 25 | 25 |
| `ibm-granite/granite-4.1-8b` | none | 0.4 | -0.560 | 25 | 25 |
| `ibm-granite/granite-4.1-8b` | none | 0.8 | -0.400 | 25 | 25 |
| `ibm-granite/granite-4.1-8b` | none | 1.2 | -0.400 | 25 | 25 |
| `ibm-granite/granite-4.1-8b` | none | 1.6 | -0.095 | 14 | 9 |
| `meta-llama/llama-3.1-8b-instruct` | adversary_bluff | 0.0 | -0.105 | 13 | 19 |
| `meta-llama/llama-3.1-8b-instruct` | adversary_bluff | 0.4 | +0.188 | 16 | 24 |
| `meta-llama/llama-3.1-8b-instruct` | adversary_bluff | 0.8 | -0.235 | 15 | 23 |
| `meta-llama/llama-3.1-8b-instruct` | adversary_bluff | 1.2 | +0.000 | 1 | 1 |
| `meta-llama/llama-3.1-8b-instruct` | adversary_bluff | 1.6 | n/a | 2 | 0 |
| `meta-llama/llama-3.1-8b-instruct` | average_person | 0.0 | n/a | 0 | 25 |
| `meta-llama/llama-3.1-8b-instruct` | average_person | 0.4 | +0.016 | 18 | 25 |
| `meta-llama/llama-3.1-8b-instruct` | average_person | 0.8 | -0.095 | 19 | 25 |
| `meta-llama/llama-3.1-8b-instruct` | average_person | 1.2 | +0.050 | 10 | 12 |
| `meta-llama/llama-3.1-8b-instruct` | average_person | 1.6 | n/a | 0 | 1 |
| `meta-llama/llama-3.1-8b-instruct` | expert | 0.0 | -0.080 | 25 | 25 |
| `meta-llama/llama-3.1-8b-instruct` | expert | 0.4 | +0.000 | 24 | 24 |
| `meta-llama/llama-3.1-8b-instruct` | expert | 0.8 | -0.077 | 24 | 25 |
| `meta-llama/llama-3.1-8b-instruct` | expert | 1.2 | +0.000 | 6 | 3 |
| `meta-llama/llama-3.1-8b-instruct` | expert | 1.6 | n/a | 0 | 0 |
| `meta-llama/llama-3.1-8b-instruct` | none | 0.0 | -0.250 | 25 | 4 |
| `meta-llama/llama-3.1-8b-instruct` | none | 0.4 | -0.167 | 23 | 18 |
| `meta-llama/llama-3.1-8b-instruct` | none | 0.8 | -0.080 | 23 | 19 |
| `meta-llama/llama-3.1-8b-instruct` | none | 1.2 | -0.123 | 13 | 5 |
| `meta-llama/llama-3.1-8b-instruct` | none | 1.6 | +1.000 | 2 | 1 |
| `openai/gpt-4o-mini` | adversary_bluff | 0.0 | +0.000 | 25 | 25 |
| `openai/gpt-4o-mini` | adversary_bluff | 0.4 | +0.080 | 25 | 25 |
| `openai/gpt-4o-mini` | adversary_bluff | 0.8 | -0.120 | 25 | 25 |
| `openai/gpt-4o-mini` | adversary_bluff | 1.2 | -0.320 | 25 | 25 |
| `openai/gpt-4o-mini` | adversary_bluff | 1.6 | n/a | 0 | 1 |
| `openai/gpt-4o-mini` | average_person | 0.0 | -0.520 | 25 | 25 |
| `openai/gpt-4o-mini` | average_person | 0.4 | -0.640 | 25 | 25 |
| `openai/gpt-4o-mini` | average_person | 0.8 | -0.680 | 25 | 25 |
| `openai/gpt-4o-mini` | average_person | 1.2 | -0.680 | 25 | 25 |
| `openai/gpt-4o-mini` | average_person | 1.6 | -0.193 | 8 | 11 |
| `openai/gpt-4o-mini` | expert | 0.0 | -0.280 | 25 | 25 |
| `openai/gpt-4o-mini` | expert | 0.4 | -0.480 | 25 | 25 |
| `openai/gpt-4o-mini` | expert | 0.8 | -0.520 | 25 | 25 |
| `openai/gpt-4o-mini` | expert | 1.2 | -0.320 | 25 | 25 |
| `openai/gpt-4o-mini` | expert | 1.6 | -1.000 | 1 | 1 |
| `openai/gpt-4o-mini` | none | 0.0 | -0.680 | 25 | 25 |
| `openai/gpt-4o-mini` | none | 0.4 | -0.640 | 25 | 25 |
| `openai/gpt-4o-mini` | none | 0.8 | -0.440 | 25 | 25 |
| `openai/gpt-4o-mini` | none | 1.2 | -0.360 | 25 | 25 |
| `openai/gpt-4o-mini` | none | 1.6 | +0.000 | 6 | 12 |
| `qwen/qwen3-30b-a3b-instruct-2507` | adversary_bluff | 0.0 | -0.200 | 10 | 10 |
| `qwen/qwen3-30b-a3b-instruct-2507` | adversary_bluff | 0.4 | +0.200 | 10 | 10 |
| `qwen/qwen3-30b-a3b-instruct-2507` | adversary_bluff | 0.8 | -0.111 | 9 | 9 |
| `qwen/qwen3-30b-a3b-instruct-2507` | adversary_bluff | 1.2 | -0.233 | 12 | 10 |
| `qwen/qwen3-30b-a3b-instruct-2507` | adversary_bluff | 1.6 | -0.212 | 8 | 13 |
| `qwen/qwen3-30b-a3b-instruct-2507` | average_person | 0.0 | -1.000 | 25 | 25 |
| `qwen/qwen3-30b-a3b-instruct-2507` | average_person | 0.4 | -1.000 | 25 | 25 |
| `qwen/qwen3-30b-a3b-instruct-2507` | average_person | 0.8 | -1.000 | 25 | 25 |
| `qwen/qwen3-30b-a3b-instruct-2507` | average_person | 1.2 | -1.000 | 25 | 25 |
| `qwen/qwen3-30b-a3b-instruct-2507` | average_person | 1.6 | -1.000 | 25 | 25 |
| `qwen/qwen3-30b-a3b-instruct-2507` | expert | 0.0 | -1.000 | 25 | 24 |
| `qwen/qwen3-30b-a3b-instruct-2507` | expert | 0.4 | -1.000 | 25 | 25 |
| `qwen/qwen3-30b-a3b-instruct-2507` | expert | 0.8 | -1.000 | 25 | 25 |
| `qwen/qwen3-30b-a3b-instruct-2507` | expert | 1.2 | -0.920 | 25 | 25 |
| `qwen/qwen3-30b-a3b-instruct-2507` | expert | 1.6 | -0.720 | 25 | 25 |
| `qwen/qwen3-30b-a3b-instruct-2507` | none | 0.0 | -1.000 | 25 | 25 |
| `qwen/qwen3-30b-a3b-instruct-2507` | none | 0.4 | -1.000 | 25 | 25 |
| `qwen/qwen3-30b-a3b-instruct-2507` | none | 0.8 | -1.000 | 25 | 25 |
| `qwen/qwen3-30b-a3b-instruct-2507` | none | 1.2 | -1.000 | 25 | 25 |
| `qwen/qwen3-30b-a3b-instruct-2507` | none | 1.6 | -1.000 | 25 | 25 |

## Non-binary answers (unconditional)

| answer | rate pooled over models [95% CI] | n cells |
|---|---|---|
| RANDOMISE | 0.0401 [0.0117, 0.0859] | 400 |
| THIRD_OPTION | 0.0008 [0.0002, 0.0014] | 400 |
| REFUSE | 0.0071 [0.0024, 0.0122] | 400 |
| UNPARSEABLE | 0.1425 [0.0321, 0.2955] | 400 |
| **excluded from plot 1** | **0.1905 [0.0708, 0.3490]** | 400 |

UNPARSEABLE: 1419 of 9992 joined cot records (0.1420 raw share); pooled over models 0.1425 [0.0321, 0.2955]. PLAN.md §4 targets under 2% and wants every instance hand-reviewed.

### UNPARSEABLE per model

| model | n UNPARSEABLE | n cot records | share |
|---|---|---|---|
| `qwen/qwen3-30b-a3b-instruct-2507` | 49 | 2000 | 0.0245 |
| `meta-llama/llama-3.1-8b-instruct` | 869 | 1992 | 0.4362 |
| `ibm-granite/granite-4.1-8b` | 219 | 2000 | 0.1095 |
| `google/gemma-4-26b-a4b-it` | 2 | 2000 | 0.0010 |
| `openai/gpt-4o-mini` | 280 | 2000 | 0.1400 |

## Direct mode

```
direct-mode records: 80
  usable (status ok):     73
  no_logprobs             0
  no_answer_tokens        7
models with a usable curve: 3 ['meta-llama/llama-3.1-8b-instruct', 'openai/gpt-4o-mini', 'qwen/qwen3-30b-a3b-instruct-2507']
  dropped google/gemma-4-26b-a4b-it: 3/16 direct records unusable (3× no_answer_tokens)
  dropped ibm-granite/granite-4.1-8b: 4/16 direct records unusable (4× no_answer_tokens)
```

| model | n direct records | ok | no_logprobs | no_answer_tokens | overlay |
|---|---|---|---|---|---|
| `google/gemma-4-26b-a4b-it` | 16 | 13 | 0 | 3 | no |
| `ibm-granite/granite-4.1-8b` | 16 | 12 | 0 | 4 | no |
| `meta-llama/llama-3.1-8b-instruct` | 16 | 16 | 0 | 0 | yes |
| `openai/gpt-4o-mini` | 16 | 16 | 0 | 0 | yes |
| `qwen/qwen3-30b-a3b-instruct-2507` | 16 | 16 | 0 | 0 | yes |

Records that could not feed the curve: **7** of 80. These failures cluster by provider and model, not by condition, so the whole model is dropped from the overlay rather than the individual cells - dropping cells would reweight the survivors.

### Analytic conditional KEEP rate, P(KEEP | K or S)

| model | scenario | T=0.0 | T=0.4 | T=0.8 | T=1.2 | T=1.6 |
|---|---|---|---|---|---|---|
| `qwen/qwen3-30b-a3b-instruct-2507` | default_keep | 0.000 | 0.000 | 0.001 | 0.009 | 0.028 |
| `qwen/qwen3-30b-a3b-instruct-2507` | default_switch | 0.500 | 0.515 | 0.543 | 0.539 | 0.533 |
| `qwen/qwen3-30b-a3b-instruct-2507` | neutral_switch | 0.000 | 0.088 | 0.128 | 0.176 | 0.221 |
| `qwen/qwen3-30b-a3b-instruct-2507` | symmetric_blind | 0.000 | 0.000 | 0.002 | 0.013 | 0.036 |
| `meta-llama/llama-3.1-8b-instruct` | default_keep | 0.000 | 0.007 | 0.073 | 0.154 | 0.217 |
| `meta-llama/llama-3.1-8b-instruct` | default_switch | 0.750 | 0.571 | 0.550 | 0.536 | 0.528 |
| `meta-llama/llama-3.1-8b-instruct` | neutral_switch | 0.250 | 0.411 | 0.452 | 0.468 | 0.476 |
| `meta-llama/llama-3.1-8b-instruct` | symmetric_blind | 0.000 | 0.058 | 0.187 | 0.269 | 0.320 |
| `openai/gpt-4o-mini` | default_keep | 1.000 | 0.995 | 0.955 | 0.908 | 0.865 |
| `openai/gpt-4o-mini` | default_switch | 1.000 | 1.000 | 0.998 | 0.989 | 0.975 |
| `openai/gpt-4o-mini` | neutral_switch | 1.000 | 1.000 | 0.987 | 0.960 | 0.925 |
| `openai/gpt-4o-mini` | symmetric_blind | 1.000 | 0.999 | 0.985 | 0.960 | 0.930 |

Averaged over persona cells with equal weight. Conditional by construction: `rescale` renormalises over {K, S} only.

## Depth distribution

Shares are per-cell, pooled with cells weighted equally and one vote per model. Every cot response has a depth code; an unparseable annotation is `NON`.

**Excluded from every pooled figure in this section: `qwen/qwen3-30b-a3b-instruct-2507`** - `contracts.NO_COT_MODELS`, which emit no visible reasoning to code (CONTRACTS.md §8). Their codes appear in the per-model table below and nowhere else. Plot 3 excludes the same models, so the figure and this table are the same quantity.

| grouping | L0 | L1 | L2 | L3 | MIX | NON |
|---|---|---|---|---|---|---|
| persona=none | 0.265 | 0.551 | 0.040 | 0.002 | 0.007 | 0.136 |
| persona=average_person | 0.309 | 0.547 | 0.018 | 0.001 | 0.007 | 0.119 |
| persona=expert | 0.147 | 0.551 | 0.113 | 0.003 | 0.026 | 0.160 |
| persona=adversary_bluff | 0.008 | 0.042 | 0.510 | 0.248 | 0.018 | 0.175 |

### Per model (never pooled - contracts.NO_COT_MODELS emit no visible reasoning, so their codes carry no information)

| model | visible cot? | L0 | L1 | L2 | L3 | MIX | NON |
|---|---|---|---|---|---|---|---|
| `qwen/qwen3-30b-a3b-instruct-2507` | no | 0.748 | 0.029 | 0.041 | 0.067 | 0.101 | 0.013 |
| `meta-llama/llama-3.1-8b-instruct` | yes | 0.178 | 0.290 | 0.083 | 0.093 | 0.017 | 0.339 |
| `ibm-granite/granite-4.1-8b` | yes | 0.186 | 0.468 | 0.169 | 0.045 | 0.027 | 0.105 |
| `google/gemma-4-26b-a4b-it` | yes | 0.233 | 0.430 | 0.218 | 0.107 | 0.009 | 0.003 |
| `openai/gpt-4o-mini` | yes | 0.133 | 0.505 | 0.209 | 0.008 | 0.005 | 0.141 |

### Flags

Pooled over codeable models only; `qwen/qwen3-30b-a3b-instruct-2507` excluded, as above - both flags are read off reasoning text.

- `flag_no_basis`: 0.138 [0.040, 0.236] pooled over 4 models.
- `flag_rigged`: 0.013 [0.006, 0.019] pooled over 4 models.

Per model, so a model with no visible reasoning is visible as such rather than averaged away:

| model | visible cot? | flag_no_basis | flag_rigged |
|---|---|---|---|
| `qwen/qwen3-30b-a3b-instruct-2507` | no | 0.129 | 0.000 |
| `meta-llama/llama-3.1-8b-instruct` | yes | 0.029 | 0.022 |
| `ibm-granite/granite-4.1-8b` | yes | 0.178 | 0.004 |
| `google/gemma-4-26b-a4b-it` | yes | 0.294 | 0.015 |
| `openai/gpt-4o-mini` | yes | 0.051 | 0.011 |

## Figures

| file | denominator |
|---|---|
| `keep_vs_temperature.png` | conditional - matches the analytic curve |
| `flip_rate.png` | bars unconditional; flip rate conditional |
| `depth_by_persona.png` | unconditional |
| `nonbinary_by_temperature.png` | unconditional |
| `answers_by_role.png` | unconditional - other answers shown, not dropped |
| `keep_by_temperature.png` | conditional - same denominator as plot 1 |

