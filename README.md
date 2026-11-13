# Two Bottles: do human biases in adversarial mind games replicate in LLMs?

**[Interactive results →](https://calkes314.github.io/two-bottles-llm-biases/)**

A man hands you one of two identical bottles. One is poisoned, and he knows which. 
You will have to drink one of them, and he wants you to drink the poisoned one.
You may keep yours or swap with his. The game is symmetric and no answer is better
than the other, so any systematic preference comes from framing or biases, not payoffs.

**Human baseline.** Over about two years I put this game to roughly 200 people in
different settings. Almost all of them kept their bottle: about 90% one-on-one with time
to think (n≈150), and two other askers replicated this at 80% and 95%. That is a
strong status quo bias. This project asks whether LLMs show the same bias.

## Questions

1. **Does the default flip the answer?** Two framings of the same game differ in one
   thing: whether doing nothing means *keeping* (`default_keep`) or *trading*
   (`default_switch`). If a model follows the default in both, that is status quo bias.
2. **Does chain-of-thought damp or amplify sensitivity to temperature?** Compare an
   analytic curve computed from direct-mode logprobs with sampled CoT answers.
3. **Does persona change reasoning depth?** Code every response by how many levels of
   opponent modelling it does ("he knows I know…", levels L0–L3).

## Design

| Factor | Levels |
|---|---|
| Models | Qwen3-30B-A3B, Llama-3.1-8B, Granite-4.1-8B, Gemma-4-26B, GPT-4o-mini |
| Framing | default_keep, default_switch (+ neutral and symmetric-blind controls) |
| Persona | none, average person, expert game theorist, adversary-who-may-bluff |
| Temperature | 0.0 – 1.6 (5 levels) |
| Replicates | 25 per cell → **~10,000 CoT responses** |

An LLM annotator (DeepSeek-V3.2, deliberately *not* one of the subject models) codes
each answer and its reasoning depth. Intervals are 95% bootstrap CIs over
(model × persona × temperature) cells, so repeated samples are never treated as
independent evidence.

## Results

![Flip rate per model](flip_rate.png)

- **Models do not show human-like status quo bias. Most show the reverse.** Flip rate
  = P(KEEP | default_keep) − P(KEEP | default_switch). Positive means the model
  follows the default. Qwen3 (−0.76), Granite (−0.50) and GPT-4o-mini (−0.41) mostly
  *act* in both framings. Only Gemma follows the default (+0.17). Pooled over models
  the estimate is −0.30 [−0.59, 0.02].
- **The adversary persona makes models reason deeper.** It moves L2+ reasoning from
  ~4% to ~76% of responses.
- **Temperature barely moves the answer.** Pooled KEEP stays within 47–54% from T=0 to
  T=1.6. Higher temperature mainly raises the share of unparseable answers.
- **The more powerful the models become, the more they seem to show the bias.** The
  medium strength models are split almost 50-50 and Claude SOTA models seem to answer
  very human-like from small tests

## Limitations

- Small and mid-tier models only. Frontier models are only spot-checked in the playground.
- 14% of answers were unparseable overall and 44% for Llama, which weakens its estimates.
- Qwen3-instruct gives no visible reasoning, so it is left out of the depth analysis.
- Total compute cost: about €7.
- The human data is informal and approximate, and the sample skews young (15–25) and
  mathematically able.

## Reproducing

All code is in [`experiment/`](experiment/), with 469 offline tests. Its
[README](experiment/README.md) documents each stage, and [`PLAN.md`](experiment/PLAN.md)
and [`CONTRACTS.md`](experiment/CONTRACTS.md) give the full specification.

```bash
cd experiment
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
export OPENROUTER_API_KEY=...
bash scripts/overnight.sh   # full resumable run: direct, contamination, CoT, annotate, analyse
```
