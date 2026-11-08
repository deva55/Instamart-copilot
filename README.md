# Instamart Customer Intelligence & Support Copilot

## Overview

This is an internal tool for the Instamart Customer Experience team, built in four parts. Part A cleans the raw synthetic data and loads them into a SQLite database. Part B segments customers and predicts who is likely to churn. Part C sorts support tickets and masks personal details. Part D answers policy questions and handles refund requests using the Part A database.

All data is synthetic and generated from `generate_data.py`. Every language-model call goes through the offline `mock_llm.py`, so no API key or internet connection is needed. All amounts are in INR.

## Reproduce

Run from the repository root, in this order:

1. `python -m venv .venv`
2. `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python part_a.py`
6. `python run_sql.py`
7. `python eda.py`
8. `python part_b.py`
9. `python part_c.py`
10. `python part_d.py`
11. `python validate.py`

Step 4 ends with `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. Step 11 prints `ALL CHECKPOINTS PASSED`. It runs after Part B because it also checks the segment file.

## Part A

### Setup

Python 3.11+ in a virtual environment. `requirements.txt` was created with `pip freeze`. `.gitignore` covers `.env`, the venv folders and `__pycache__/`.

### Raw data audit

**customers.csv**: 1,200 rows, 5 columns

| Column | Type | Missing | Missing % |
|---|---|---|---|
| customer_id | str | 0 | 0.00 |
| city | str | 0 | 0.00 |
| signup_date | str | 0 | 0.00 |
| age | float64 | 61 | 5.08 |
| gender | str | 30 | 2.50 |

**products.csv**: 40 rows, 4 columns. No missing values. product_id, product_name and category are str, price_inr is int64.

**orders.csv**: 13,084 rows, 9 columns

| Column | Type | Missing | Missing % |
|---|---|---|---|
| order_id | str | 0 | 0.00 |
| customer_id | str | 0 | 0.00 |
| product_id | str | 0 | 0.00 |
| quantity | int64 | 0 | 0.00 |
| order_ts | str | 0 | 0.00 |
| status | str | 0 | 0.00 |
| discount_pct | int64 | 0 | 0.00 |
| order_amount | str | 53 | 0.41 |
| delivery_minutes | float64 | 2,541 | 19.42 |

The 53 missing amounts are actually the text `N/A`, which pandas reads as missing. Along with 67 `error` values they make up the 120 non-numeric amounts below.

| Check | Count |
|---|---|
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Pending rows | 656 |
| Distinct order_id | 12,309 |
| Non-numeric order_amount | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Blank age | 61 |
| Age outside 10–100 | 6 |
| Raw city spellings | 24 |
| Non-blank raw gender spellings | 10 |
| Blank gender | 30 |

### Cleaning rules

**Customers**

| Rule | Changed |
|---|---|
| City: trim spaces, title case | 476 rows |
| Gender: map to male / female, blank to unknown | 950 rows |
| Ages outside 10–100 set to missing | 6 values |
| Missing ages filled with the city median (groupby + transform), rounded | 67 values |
| signup_date converted to datetime | 1,200 rows |

After cleaning: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Female 610, male 560, unknown 30. No missing ages, all between 18 and 60.

**Orders** (in this order)

| Step | Rule | Changed |
|---|---|---|
| 1 | order_ts converted to datetime | all rows |
| 2 | Exact duplicates removed | 119 rows (13,084 → 12,965) |
| 3 | Kept only the latest order_ts per order_id | 656 rows (12,965 → 12,309) |
| 4 | order_amount made numeric, bad values filled with round(price × quantity × (1 − discount_pct/100), 2) | 112 values |
| 5 | Negative delivery_minutes set to missing | 105 values |
| 6 | Missing delivery_minutes of completed orders filled with the category median among completed orders (groupby + transform) | 774 values |
| 7 | Cancelled orders left without a delivery time, since they were never delivered | 1,199 rows |

Steps 4 and 5 change fewer values than the audit found (112 vs 120, 105 vs 106) because some of those sat on rows already dropped in steps 2 and 3. The clean orders table also keeps `price_inr` and `product_category`, which are used later.

After cleaning: 12,309 rows, 11,110 completed and 1,199 cancelled, no pending rows, no missing amounts, no missing delivery times on completed orders, completed revenue INR 2,998,668.80.

### Storage and validation

Clean tables are saved to `data/clean/` as CSV and to `data/instamart.db` as `customers`, `products` and `orders`, with `order_ts` stored as text (`YYYY-MM-DD HH:MM:SS`). `validate.py` asserts the audit and cleaning checkpoints and prints `ALL CHECKPOINTS PASSED`.

### SQL analysis

Queries are in `sql/queries.sql` and run by `run_sql.py`. Only completed orders count as revenue.

| Query | Result |
|---|---|
| Q1 WHERE + HAVING | 30 customers |
| Q2 Revenue by city and category | 30 rows, total INR 2,998,668.80, largest Bengaluru / Electronics INR 233,909.45 |
| Q3 Above-average price | 11 products |
| Q4 No completed order | 25 customers |
| Q5 Above-average spend | 550 customers, average INR 2,552.06 |
| Q7 Running total | 183 days, final value INR 2,998,668.80 |

**Q6** top customer per city: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10).

8 rows have a different RANK and ROW_NUMBER, because customers with the same total spend share a rank while ROW_NUMBER still numbers them one by one.

### pandas cross-check and EDA

The Q2 pivot table (`margins=True`) gives a grand total of INR 2,998,668.80, the same as SQL.

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

| Completed order_amount | Value |
|---|---|
| Mean / median | INR 269.91 / INR 135.00 |
| Skewness | 4.61 |
| Q1 / Q3 | INR 64.80 / INR 297.00 |
| Lower fence | INR −283.50, 0 outliers |
| Upper fence | INR 645.30, 897 outliers |

**What is this data about?** About 12,300 synthetic orders from 1,200 customers in six cities, April to September 2025.

**What do the distributions look like?** Order value is heavily right-skewed. The mean is double the median and skewness is 4.61. Most orders are small grocery, dairy and snack baskets, while electronics brings in about 41% of revenue.

**Outliers or anomalies?** 897 orders are above the upper fence and none below. About 70% of them are electronics, so they are real high-value orders, not errors, and I kept them. The real anomalies (duplicates, bad amounts, negative delivery times) were fixed in cleaning.

**Relationships?** All correlations between order_amount, discount_pct, quantity and delivery_minutes are weak. The strongest is order_amount with discount_pct at −0.095, just ahead of order_amount with quantity at +0.087. A correlation this small does not show causation.

## Part B

Everything is in `part_b.py`. Customer behaviour uses completed orders only.

### RFM

Snapshot: 2025-10-01 00:00:00.

| Measure | Value |
|---|---|
| Customers | 1,175 |
| Recency median / min / max | 19 / 0 / 180 days |
| Frequency mean / max | 9.46 / 33 |
| Monetary mean / max | INR 2,552.06 / INR 14,766.55 |

### K-means

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.479 |
| 6 | 503.2 | 0.482 |
| 7 | 408.7 | 0.419 |
| 8 | 367.2 | 0.410 |

Inertia drops sharply up to K=4 and then flattens, so the elbow is at 4. K=4 also has the best silhouette score, and four groups are easy for the retention team to act on. I chose **K = 4**.

K=8 always has lower inertia than K=2 because more centroids means every customer ends up closer to one of them, and with one cluster per customer inertia would hit zero. So lower inertia alone just reflects more clusters, not better ones, and can't be used to pick K.

### Segments

| Segment id | Name | Size | Recency (days) | Frequency | Monetary (INR) | Action |
|---|---|---|---|---|---|---|
| 0 | Premium | 138 | 24.1 | 6.8 | 7,129.70 | Priority support and early access to new electronics |
| 1 | Lapsing | 182 | 109.0 | 2.2 | 685.99 | Win-back discount on their next order |
| 2 | Active | 419 | 8.8 | 19.4 | 3,410.02 | Loyalty rewards on everyday groceries |
| 3 | Occasional | 436 | 28.9 | 3.8 | 1,057.62 | Bundles and reminders to build a habit |

Premium is the segment with the highest mean monetary value. Segment ids are just labels with no order.

PCA explained variance ratio: 0.634 and 0.218, total 0.852.

Saved to `outputs/customer_segments.csv`.

### Lapse problem

Cutoff 2025-09-01. The cohort is every customer with a completed order before the cutoff. Lapsed = 1 if they have no completed order from 2025-09-01 to 2025-09-30.

Features (all from before the cutoff): recency, frequency, monetary, average discount, average delivery time, cancellation rate, favourite category, city, age, gender and tenure in days.

| Measure | Value |
|---|---|
| Cohort | 1,156 |
| Lapsed | 443 (38.32%) |
| Train / test | 924 / 232 |
| Baseline (always "not lapsed") | Accuracy 61.6%, lapsed F1 0 |

### Preprocessing

Numeric features are scaled with StandardScaler and city, gender and favourite category are one-hot encoded with `drop="first"`, all inside a Pipeline fitted on training rows only. No feature column has missing values (checked in code).

The scaler's learned mean of frequency is 8.11, against 8.03 for the full cohort. They differ because the scaler only saw the training rows.

### Models

| Model | Best parameters | Best CV F1 |
|---|---|---|
| Logistic regression | C = 100 | 0.651 |
| Random forest | max_depth = 5, min_samples_leaf = 5 | 0.681 |

Test set, lapsed class:

| Model | Confusion matrix [[TN, FP], [FN, TP]] | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[114, 29], [33, 56]] | 0.659 | 0.629 | 0.644 |

I picked **logistic regression**. It has the higher F1 and recall, catching 58 of 89 lapsing customers against 56. Missing a customer who is about to leave loses all their future orders, while sending an offer to a loyal customer only costs a voucher, so recall matters more here than the forest's small edge in precision. It is also easier to explain to the business.

### Class balance

I over-sampled the training set only (216 duplicate lapsed rows, random_state=42, 570 of each class) and refitted the logistic regression.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Original | 0.644 | 0.652 | 0.648 |
| Balanced | 0.589 | 0.854 | 0.697 |

Recall moved the most, up 0.20. The balanced model catches 76 of 89 lapsing customers instead of 58, but precision drops a little, so the team would also send more offers to customers who were going to stay anyway.

### Interpretation

Top 5 random forest features:

| Feature | Importance | Meaning |
|---|---|---|
| frequency | 0.305 | Customers who order often keep ordering |
| monetary | 0.279 | Total spend, closely linked to how often they order |
| recency | 0.113 | A long gap since the last order is a warning sign |
| cancellation_rate | 0.072 | Frequent cancellations point to a bad experience |
| avg_discount | 0.048 | Discount use has only a small effect |

Logistic regression signs:

| Feature | Sign | Meaning |
|---|---|---|
| frequency | Negative | More orders, less likely to lapse |
| monetary | Positive | For the same number of orders, big spenders who order rarely are slightly more at risk |
| recency | Positive | The longer since the last order, the higher the risk |

The model is saved to `outputs/lapse_model.joblib`. Reloading it gives exactly the same test predictions.

**Retention list.** 40 test customers are Premium. The top 10 by lapse probability are in `outputs/retention_list.csv`, and 3 of them actually lapsed. The Premium flag uses the whole-window RFM, which includes September, so this list is illustrative and not a leakage-free evaluation.

**Leakage.** No encoder, scaler or resampler saw the test rows. No data from 2025-09-01 onwards was used in the features. The segment labels were not used as features.

## Part C

Everything is in `part_c.py`. Tickets are processed in file order, and every model call goes through `mock_llm.py`.

### Input guardrails

Before any model call, empty tickets get `skipped_empty` and tickets over 400 estimated tokens get `flagged_long`. Neither is ever sent to the model.

| Check | Tickets |
|---|---|
| skipped_empty | 4 |
| flagged_long | 4 (734 tokens each) |
| Sent to the model | 156 |

### Redaction

Emails, phone numbers and order ids are replaced in that order with `[EMAIL]`, `[PHONE]` and `[ORDER_ID]` across all 164 tickets.

| Type | Substitutions |
|---|---|
| Email | 55 |
| Phone | 91 |
| Order id | 132 |
| Total | 278 |

Re-applying the three patterns to the redacted texts finds 0 matches. The user message sent to the model is only the redacted ticket text, and an assertion before every call checks that no email, phone or order id is in it.

### Prompts

Both prompts contain CLASSIFY, list the four categories and ask for JSON only. The zero-shot prompt is 86 tokens. The few-shot prompt is 142 tokens and adds an `EXAMPLES:` line with a billing and a fraud example.

### Retries

Every call is wrapped in a retry with up to 3 retries per ticket. The wait before retry n is 0.05 × 2^n × a random factor between 0.9 and 1.15, and each retry prints the ticket id, retry number and wait. `mock_llm.reset()` runs before each full run.

A full run over the 156 tickets hit 19 TransientLLMErrors. All 19 were recovered on the first retry (retry 0), with waits between 0.0451 and 0.0575 seconds, and no ticket ended with a permanent error.

### Output validation

A reply is accepted as `classified_json` if it is valid JSON with an allowed category. If it isn't JSON, it is accepted as `classified_keyword` only when exactly one category word appears in it (words split with `re.findall(r"[a-z]+", ...)`). Everything else, including valid JSON with a category outside the list, is `flagged_invalid_output` and goes to human review.

| Status | Tickets |
|---|---|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |

### Zero-shot vs few-shot

| Run | Classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126 of 147) | 17,824 | 121.3 |
| Few-shot | 147 | 100% (147 of 147) | 26,560 | 180.7 |

Zero-shot confusion table (rows are the true category):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

The few-shot prompt costs about 59 extra tokens per classified ticket (180.7 vs 121.3), or 8,736 more tokens for the whole run. For that it fixes all 21 zero-shot mistakes, which works out to about 416 extra tokens per corrected ticket. The mistakes that matter most are the 11 fraud tickets read as billing, because they would go to the normal support queue instead of the security team. So the extra tokens are worth paying whenever a wrong category sends a ticket to the wrong team, which is clearly the case for fraud.

### Budget and routing

**Budget demo.** Zero-shot run with TOKEN_BUDGET = 6000, saved to `outputs/triage_budget_demo.csv`. The run used 6,058 tokens and **103 tickets** ended `budget_exhausted` and went to human review.

**Final output.** Few-shot run with TOKEN_BUDGET = 100000 (26,560 tokens used, nothing exhausted), routed with the Part B segment file and saved to `outputs/triage_results.csv` (164 rows).

| Route | Priority | Tickets |
|---|---|---|
| support_queue | NORMAL | 102 |
| security_team | HIGH | 34 |
| human_review | NORMAL | 17 |
| premium_desk | HIGH | 11 |

**Rule-validated vs human.** 147 tickets were classified and passed the output validation rules (135 as JSON, 12 by keyword), and were routed automatically. The other 17 went to human review: 4 empty, 4 too long and 9 with invalid model output.

## Part D

Everything is in `part_d.py`. All model calls go through `mock_llm.py`.

### Chunking

Each policy file is split into 40-word chunks with a 10-word overlap, and every chunk keeps its source file name.

| File | Chunks |
|---|---|
| refund_policy.txt | 4 |
| delivery_policy.txt | 3 |
| escalation_policy.txt | 3 |
| payment_policy.txt | 3 |
| privacy_policy.txt | 2 |
| Total | 15 |

### Retrieval

`TfidfVectorizer()` is fitted on the chunk texts, the query is transformed with the same vectorizer, and chunks are ranked by cosine similarity. `retrieve(query, k=3)` returns source, score and text. For "refund policy late delivery" the top three chunks all come from refund_policy.txt (scores 0.272, 0.251, 0.245).

TF (term frequency) measures how often a word appears in a chunk, and IDF (inverse document frequency) measures how rare that word is across all chunks, so common words count for less. The limitation is that TF-IDF only matches exact words, so a question like "money back for a slow order" can miss the chunk that says "refund" and "late delivery" even though they mean the same thing.

### Answering

`answer(question)` retrieves three chunks and sends two messages to the model: a POLICY ASSISTANT system message telling it to answer only from the context (and say it could not find the information otherwise), and a user message in the form `CONTEXT:` / `[source]` + chunk ×3 / `QUESTION:` / question. It returns the answer and the sources used.

### Evaluation

| Check | Overlap 10 | Overlap 0 |
|---|---|---|
| Chunks | 15 | 13 |
| expected_doc in top 3 (12 answerable) | 12/12 | 12/12 |
| expected_fact in answer (12 answerable) | 12/12 | 10/12 |
| Fallback answer (2 unanswerable) | 2/2 | 2/2 |

With overlap 0, two questions break:

- "What delivery time makes a completed order eligible for a full refund?" The answer comes back as "receives a full refund of the order amount." without "more than 45 minutes".
- "How soon is money returned when a payment fails but is deducted?" The answer stops at "the amount is returned automatically" without "within 48 hours".

In both cases the right document is still retrieved, but the sentence is cut in half at a 40-word chunk boundary, so the chunk the model answers from doesn't contain the fact. The 10-word overlap keeps each of these sentences whole in at least one chunk.

### Tools

Five tools, each with a name, a one-sentence description and a parameter spec in `TOOLS_SPEC`, and looked up by name in `TOOL_MAP`:

- `get_order` reads `data/instamart.db` and returns order_id, status, order_amount and delivery_minutes, or `{"error": "order not found"}`
- `search_policy` returns the `retrieve` results
- `calculate_refund` returns 100% of the amount if delivery took more than 45 minutes, otherwise 0
- `issue_refund` and `escalate_to_human` each add one line (timestamp, tool, order id, amount) to `outputs/refund_ledger.jsonl`

### Agent results

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,132 |
| S2 | REFUND | 135.00 | 5 | 1,132 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,119 |
| S4 | DENIED | 0.00 | 4 | 793 |
| S5 | DENIED | 0.00 | 3 | 478 |
| S6 | NOT_FOUND | 0.00 | 2 | 166 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,282 |

Every run is logged in `outputs/agent_runs.jsonl` with all its steps (tool, arguments, observation). S7 uses all six model calls with the default token limit of 6,000.

### Stopping safely

| Demo | Status | Model calls |
|---|---|---|
| S1 with token_limit=200 | FALLBACK_TOKEN_LIMIT | 3 (stopped at 495 tokens, before running calculate_refund) |
| S1 with time_limit_seconds=0 | FALLBACK_TIMEOUT | 0 (stopped before the first call) |

Every fallback returns the steps done so far and a short handoff summary for a human.

### Crash and resume

A checkpoint is saved to `state/` after every tool observation. For the demo, S1 crashes on purpose right after the second tool, then `resume_agent` loads the checkpoint and carries on.

Tools executed, in order:

1. get_order (before the crash)
2. search_policy (before the crash)
3. calculate_refund (after resume)
4. issue_refund (after resume)

Four executions, none repeated, final status REFUND with INR 128.25. Because this demo issues S1's refund a second time, the ledger ends with 4 lines.

### ReAct vs plan-and-execute

My ReAct agent makes one model call per step: 5 for S1, 3 for S5 and 2 for S6. Every call resends the whole history, which is why S1 costs 1,132 tokens. A plan-and-execute agent would ask for the full step list once, so a normal refund like S1 would need about two calls instead of five and far fewer tokens.

The difference shows up when something unexpected happens. In S6, get_order returns "order not found". ReAct sees that and stops after 2 calls. A plan would have the rest of the steps lined up, so it needs to notice the error and replan, which adds a call. S5 is harder, because a cancelled order is not an error. The plan could carry on to calculate_refund and issue_refund unless the replanning rule also checks the order status. ReAct just reads the observation and denies the refund at step 3.

For this refund workflow I would keep ReAct. The runs are short (2 to 6 calls), the token cost is small, and every next step depends on what the last tool returned. Since the agent can move money, I'd rather pay a few extra calls than risk following a plan built before the order was looked up. Plan-and-execute makes more sense for long, fixed workflows at high volume.

## Limitations and Next Steps

- The Premium flag is a post-hoc whole-window label that includes the September window used for the lapse label, so the retention list is illustrative, not a leakage-free evaluation.