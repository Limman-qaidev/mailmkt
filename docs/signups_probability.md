# Customer 360º — Signup Probability Model

**Audience:** Marketing and data teams  
**Purpose:** Explain the *current implemented* per‑recipient signup probability used in **Campaign Planning**.

---

## 1) Output of the model

For each **email** and **topic** (product/campaign theme), we estimate the probability that the **next send** will result in a **signup**:

\[
p_{\text{signup}} \equiv P(\text{signup on the next send} \mid \text{email},\ \text{topic})
\]

This is a **per‑send** probability and is used to rank/filter recipients and to estimate expected signups of an audience.

---

## 2) Data used and counting rules

All event counts are **unique per message** (`msg_id`) to avoid double counting multiple events on the same send.

For each `(email, topic)` we compute:

| Symbol | Meaning (within the topic) |
|:--|:--|
| `S` | Unique **sends** |
| `C` | Unique **clicks** |
| `Y` | **Signup indicator** (0/1). It is **1** if there is at least one signup **either** in the events stream (`event_type == "signup"`) **or** in the `signups` table after mapping `campaign → topic`; otherwise **0**. |

**Topic** is obtained from the campaigns table (column `topic` if present); otherwise it is heuristically parsed from `subject`/`name` (e.g., prefix in brackets `[Mortgage] …`, or the text before `—`, `-`, or `:`).  
Emails are normalized to lowercase and trimmed.

---

## 3) Topic priors (corpus‑level baselines)

Before individual smoothing we compute **priors** at the **topic** level:

- **Click per send prior**
  \[
  p_{0,\text{click}} = \frac{\text{unique clicks in topic}}{\text{unique sends in topic}}
  \]

- **Signup given click prior**
  \[
  p_{0,\text{sc}} = \frac{\text{unique signups in topic}}{\text{unique clicks in topic}}
  \]
  The signup numerator is the **union** of (a) signup events and (b) signups table mapped to the topic (no double counting).

For numerical stability we clamp priors to \([10^{-4},\ 0.9]\).

---

## 4) Empirical‑Bayes smoothing (per‑recipient, per‑topic)

We stabilize sparse histories by blending personal data with topic priors.

- **Smoothed click per send**
  \[
  \hat{p}_{\text{click}} = \frac{C + \alpha_C \cdot p_{0,\text{click}}}{S + \alpha_C}
  \]
  with default \(\alpha_C = 2.0\).

- **Smoothed signup given click**
  \[
  \hat{p}_{\text{sc}} = \frac{Y + \alpha_{SC} \cdot p_{0,\text{sc}}}{C + \alpha_{SC}}
  \]
  with default \(\alpha_{SC} = 2.0\).

Both are standard beta‑binomial style shrinkage estimators.

---

## 5) Final probability (current implementation)

The **per‑send** signup probability is the product of the two smoothed stages:

\[
\boxed{ \; \hat{p}_{\text{signup}} \;=\; \hat{p}_{\text{click}} \times \hat{p}_{\text{sc}} \; }
\]

Interpretation: “Chance this recipient clicks the next email” × “Chance they sign up *given* they click.”

---

## 6) Eligibility & filters (business rules)

The Campaign Planning UI applies the following rules when building audiences:

1. **Goal: New acquisition** → Exclude recipients already **registered** in this topic (“owners”), detected exactly like `Y` but without per‑send restriction.
2. **Unsubscribe / complaint in topic** → Exclude.
3. **Minimum exposure** → Require `S ≥ S_min` (default `1` in UI).
4. **Recency in topic (optional)** → Keep only emails whose **last open in this topic** is within *N* days (disabled by default).
5. **Domain filter (optional)** → Include only selected email domains.
6. **Minimum probability (optional)** → Keep only recipients with \(\hat{p}_{\text{signup}} \ge \tau\) (slider).
7. **Top‑N cut** → Keep the top *N* recipients by \(\hat{p}_{\text{signup}}\).

**Expected signups** for a built audience is the sum of member probabilities:
\[
\mathbb{E}[\text{signups}] = \sum_{i \in \text{audience}} \hat{p}_{\text{signup}, i}
\]

---

## 7) Default parameters & safeguards

- Smoothing: \(\alpha_C = \alpha_{SC} = 2.0\).
- Priors clamped to \([10^{-4}, 0.9]\) to avoid zero/one pathologies.
- Counts are **unique by `msg_id`**; timestamps are coerced to UTC when needed.
- If `S = 0` the recipient is dropped by the `S_min` exposure filter.
- If `C = 0`, \(\hat{p}_{\text{sc}}\) shrinks to the prior \(p_{0,\text{sc}}\).

---

## 8) Worked numeric examples

Assume a topic where:
- \(p_{0,\text{click}} = 0.020\) (2% clicks per send)
- \(p_{0,\text{sc}}    = 0.050\) (5% signup given click)
- \(\alpha_C = \alpha_{SC} = 2.0\)

### Example A — Some engagement, no signup yet
`S = 4`, `C = 1`, `Y = 0`

- \(\hat{p}_{\text{click}} = \frac{1 + 2\cdot0.02}{4 + 2} = \frac{1.04}{6} \approx 0.173\)
- \(\hat{p}_{\text{sc}}    = \frac{0 + 2\cdot0.05}{1 + 2} = \frac{0.10}{3} \approx 0.033\)
- \(\hat{p}_{\text{signup}} \approx 0.173 \times 0.033 = 0.0057\) → **0.57%** (≈ 5.7 per 1,000 sends)

### Example B — No clicks yet
`S = 3`, `C = 0`, `Y = 0`

- \(\hat{p}_{\text{click}} = \frac{0 + 2\cdot0.02}{3 + 2} = 0.008\)
- \(\hat{p}_{\text{sc}}    = \frac{0 + 2\cdot0.05}{0 + 2} = 0.050\)
- \(\hat{p}_{\text{signup}} = 0.008 \times 0.050 = 0.0004\) → **0.04%** (≈ 0.4 per 1,000 sends)

### Example C — Strong engagement
`S = 8`, `C = 3`, `Y = 1`

- \(\hat{p}_{\text{click}} = \frac{3 + 2\cdot0.02}{8 + 2} = 0.304\)
- \(\hat{p}_{\text{sc}}    = \frac{1 + 2\cdot0.05}{3 + 2} = 0.220\)
- \(\hat{p}_{\text{signup}} \approx 0.304 \times 0.220 = 0.0669\) → **6.69%** (≈ 66.9 per 1,000 sends)

> In **New acquisition**, emails with prior signup in the topic are excluded from targeting (even if their probability is high).

---

## 9) Implementation steps (exact)

1. Normalize emails (`str.lower().strip()`), ensure `event_ts` is UTC, and map `campaign → topic`.
2. Build topic priors \(p_{0,\text{click}}, p_{0,\text{sc}}\) from unique clicks/sends and unioned signups.
3. For each `(email, topic)` compute `S`, `C`, and `Y` (union event+table, collapsed to 0/1).
4. Compute \(\hat{p}_{\text{click}}\) and \(\hat{p}_{\text{sc}}\) with \(\alpha_C=\alpha_{SC}=2.0\).
5. Compute \(\hat{p}_{\text{signup}} = \hat{p}_{\text{click}} \times \hat{p}_{\text{sc}}\).
6. Apply business rules: owners, unsub/complaint exclusions, `S_min`, optional recency, domains, min‑probability, and top‑N cut.
7. Sum \(\hat{p}_{\text{signup}}\) across the kept recipients to get expected signups.

---

## 10) Glossary

- **Topic**: campaign/product theme (e.g., “Mortgage”, “Credit Card”).  
- **Owner**: an email that already has a signup in the topic (from events or signups table).  
- **Empirical‑Bayes**: shrinkage technique to stabilize per‑user rates using corpus‑level priors.  
- **Per‑send probability**: probability that *one additional email send* yields a signup.