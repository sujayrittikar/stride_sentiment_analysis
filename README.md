# STRIDE: Stimulus-guided Training for Reward Integration with Directional Exponential Smoothing

A reinforcement learning framework for financial aspect-based sentiment analysis (ABSA) using policy gradient optimization with EMA-smoothed REINFORCE.

## How It Works

- **Policy Agent (T5 Base)** generates top-*k* domain-relevant keyword stimuli conditioned on a headline and target aspect
- **Reward Evaluator (LLaMA 3.2 1B, frozen)** classifies aspect-level sentiment on keyword-augmented inputs and returns F1-based rewards
- **REINFORCE + EMA** optimizes keyword selection via policy gradients; EMA smoothing stabilizes the reward signal and reduces gradient variance
- **DSP (Directional Stimulus Prompting)** injects selected keywords directly into the evaluator prompt for aspect-focused reasoning

## Usage

1. Use the uploaded XML data files.
2. Open `stride_absa.ipynb` and run cells.
3. Adjust hyperparameters in the notebook as needed.

## Results

| Dataset | Accuracy | F1 | Macro-F1 |
|---|---|---|---|
| SEntFiN 1.0 | 0.950 | 0.946 | 0.935 |
| FinEntity | 0.942 | 0.933 | 0.923 |

STRIDE achieves state-of-the-art F1 on FinEntity (+4.2% over prior work) and SEntFiN 1.0 (0.946).

---

## Appendix

### A. Instruction Prompts

#### Policy Model Prompt (T5 Base)

```
Extract up to <k> concise, domain-relevant keyword phrases (1–3 words)
for the ASPECT in the PASSAGE. Return ONLY the phrases separated by ' | '.
ASPECT: <aspect> PASSAGE: <headline> Keywords:
```

#### Reward Model Prompt (LLaMA 3.2 1B)

```
You are an expert financial aspect-based sentiment analyst.
TASK: Determine the sentiment POLARITY of the SPECIFIED ASPECT in the headline.
OUTPUT FORMAT: respond with exactly one lowercase word from <positive, negative, neutral>
— no punctuation, no explanations.

DEFINITIONS (financial context):
- positive: outcomes that improve value/health/outlook (e.g., beat/raise, growth, upgrade, profit, approval, favorable guidance).
- negative: outcomes that harm value/health/outlook (e.g., miss/cut, decline, downgrade, loss, lawsuit, layoffs, guidance lowered).
- neutral: mixed/ambiguous signals; descriptive without clear direction; conditional language.

DECISION RULES:
1) Judge polarity for the ASPECT ONLY.
2) PRIORITY: headline text > aspect definition > stimulus keywords. Ignore keywords conflicting with headline.
3) Handle NEGATION & MODIFIERS precisely ('not', 'no longer', 'despite', 'only', 'barely').
4) UNCERTAINTY words ('may', 'could', 'mulls'): prefer neutral unless strong directional evidence.
5) NUMERIC & EVENT cues: 'beats/above/raises' → positive; 'miss/below/cuts' → negative.
6) MIXED signals: choose neutral unless one side clearly dominates.
7) Ignore unrelated entities unless they directly affect the specified aspect.
8) Insufficient evidence: answer neutral.

Aspect: <aspect> Headline: <headline> Stimulus keywords: <kws> Answer:
```

---

### B. Class-wise Results

**SEntFiN 1.0**

| Class | Accuracy | F1 |
|---|---|---|
| Positive | 95.67 | 95.13 |
| Negative | 95.14 | 94.32 |
| Neutral | 94.20 | 94.40 |

**FinEntity**

| Class | Accuracy | F1 |
|---|---|---|
| Positive | 93.62 | 92.04 |
| Negative | 94.07 | 95.22 |
| Neutral | 94.93 | 93.12 |

---

### C. Keyword Importance Analysis

Top-10 keywords learned by STRIDE per sentiment class, ranked by mean delta-margin (average difference between the correct class logit and the max competing logit when the keyword appears).

| Positive | Score | Negative | Score | Neutral | Score |
|---|---|---|---|---|---|
| Definance Technologies | 0.1115 | set | 0.2392 | offsets | 0.2257 |
| days mitesh | 0.1016 | immediate | 0.2103 | offer | 0.1766 |
| war | 0.0844 | firms like | 0.1559 | week | 0.1607 |
| upward | 0.0807 | wary | 0.1545 | 2014 | 0.1494 |
| resume | 0.0699 | war | 0.1183 | q1 fy13 | 0.1487 |
| experts | 0.0671 | placed | 0.1015 | NSEL | 0.1457 |
| sensex ends | 0.0571 | rupee | 0.0918 | good investment | 0.1447 |
| ends | 0.0524 | support | 0.0758 | asset | 0.1436 |
| promising | 0.0479 | rice | 0.0649 | q3 net | 0.1422 |
| 20 crore | 0.0355 | ends | 0.0423 | pact | 0.1420 |

**Observations:**
- **Positive**: Performance-oriented language and company names (e.g., "upward," "promising"); scores in the 0.05–0.11 range.
- **Negative**: Highest absolute scores (0.04–0.24); action-oriented and cautionary terms dominate (e.g., "set," "immediate," "wary").
- **Neutral**: Temporal references and descriptive financial terminology (e.g., "2014," "q1 fy13," "asset").
