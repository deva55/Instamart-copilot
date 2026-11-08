import pandas as pd
from pathlib import Path
from mock_llm import est_tokens
import re
import random
import time
import mock_llm
from mock_llm import TransientLLMError
import json

# ---------------- Step 1: load tickets and input guardrails ----------------
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

MAX_INPUT_TOKENS = 400

# keep_default_na=False so empty ticket texts stay as "" instead of NaN
tickets = pd.read_csv(RAW_DIR / "tickets.csv", keep_default_na=False)


def input_guardrail(text):
    """Return a status if the ticket must not go to the model, otherwise None."""
    if text.strip() == "":
        return "skipped_empty"
    if est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"
    return None


tickets["guardrail_status"] = tickets["text"].apply(input_guardrail)
tickets["est_tokens"] = tickets["text"].apply(est_tokens)
eligible = tickets[tickets["guardrail_status"].isna()]

# ---------------- Step 1 checks ----------------
print("Tickets loaded:", tickets.shape)
print(tickets["guardrail_status"].value_counts(dropna=False))
print(tickets.loc[tickets["guardrail_status"] == "flagged_long", ["ticket_id", "est_tokens"]].to_string(index=False))
print("Eligible for the model:", len(eligible))

# ---------------- Step 2: redact personal information ----------------
PII_PATTERNS = {
    "email": (r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),
    "phone": (r"\b[6-9]\d{9}\b", "[PHONE]"),
    "order_id": (r"ORD-\d{6}", "[ORDER_ID]")
}


# Function to redact one ticket
def redact_pii(text):

    redacted_text = text
    counts = {"email": 0, "phone": 0, "order_id": 0}

    for name, (pattern, replacement) in PII_PATTERNS.items():

        redacted_text, count = re.subn(
            pattern, replacement, redacted_text
        )

        counts[name] = count

    return redacted_text, counts


# Store redacted ticket texts
redacted_texts = []

# Total substitution counts across all tickets
total_counts = {"email": 0, "phone": 0, "order_id": 0}

# Process tickets one by one
for text in tickets["text"]:

    redacted_text, counts = redact_pii(text)

    redacted_texts.append(redacted_text)

    for name in total_counts:
        total_counts[name] += counts[name]


# Add redacted text to dataframe
tickets["redacted_text"] = redacted_texts

# ---------------- Step 2 checks ----------------
print("Emails:", total_counts["email"])
print("Phones:", total_counts["phone"])
print("Order IDs:", total_counts["order_id"])
print("Total:", sum(total_counts.values()))


#------step 3: define prompts for zero-shot and few-shot classification ------
ZERO_SHOT_PROMPT = """
CLASSIFY the customer support ticket into exactly one category.

Allowed categories:
billing, delivery, technical, fraud

Return ONLY a valid JSON object in this format:
{"category": "...", "reason": "..."}

The category must be one of the four allowed categories.
The reason must explain why the classification.
Do not invent information. If you cannot find the category, say so and it should be passed to the human validation.
"""

FEW_SHOT_PROMPT = """
CLASSIFY the customer support ticket into exactly one category.

Allowed categories:
billing, delivery, technical, fraud

Return ONLY a valid JSON object in this format:
{"category": "...", "reason": "..."}

The category must be one of the four allowed categories.
The reason must explain why the classification.
Do not invent information. If you cannot find the category, say so and it should be passed to the human validation.
Refer the below example for sample output in JSON format.

EXAMPLES:
Ticket: I was charged twice for the same order.
Response: {"category": "billing", "reason": "Duplicate payment charge."}

Ticket: Someone made an unauthorized transaction using my account.
Response: {"category": "fraud", "reason": "Potential unauthorized transaction."}
"""
print("Zero-shot prompt tokens:", est_tokens(ZERO_SHOT_PROMPT))
print("Few-shot prompt tokens:", est_tokens(FEW_SHOT_PROMPT))
print("EXAMPLES: in zero-shot:", "EXAMPLES:" in ZERO_SHOT_PROMPT)
print("EXAMPLES: in few-shot:", "EXAMPLES:" in FEW_SHOT_PROMPT)


# ---------------- Step 4: resilient model calls ----------------
BASE_DELAY = 0.05
MAX_RETRIES = 3

random.seed(42)          # makes the jitter waits repeatable
transient_errors = 0     # TransientLLMErrors raised in the current run


def call_llm(ticket_id, redacted_text, system_prompt):
    global transient_errors

    # No email, phone or order id may reach the model
    for pattern, replacement in PII_PATTERNS.values():
        assert re.search(pattern, redacted_text) is None, "PII found in prompt"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": redacted_text}
    ]

    for attempt in range(MAX_RETRIES + 1):

        try:
            response = mock_llm.chat(messages)
            return response

        except TransientLLMError:
            transient_errors += 1

            if attempt == MAX_RETRIES:
                print(f"{ticket_id}: Maximum retries exceeded")
                return None

            wait = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)

            print(
                f"Ticket {ticket_id} | "
                f"Retry {attempt} | "
                f"Wait {wait:.4f} seconds"
            )

            time.sleep(wait)


# ---------------- Step 5: validate every answer ----------------
ALLOWED_CATEGORIES = ["billing", "delivery", "technical", "fraud"]


def validate_reply(reply_text):
    """Return (status, category) for one model reply."""

    # 1. Try to read the reply as JSON
    try:
        data = json.loads(reply_text)
    except (json.JSONDecodeError, TypeError):
        data = None

    if data is not None:
        if isinstance(data, dict) and data.get("category") in ALLOWED_CATEGORIES:
            return "classified_json", data["category"]
        # Valid JSON but the category is not allowed: never guess
        return "flagged_invalid_output", None

    # 2. Not JSON: look for exactly one allowed category word
    words = re.findall(r"[a-z]+", reply_text.lower())
    found = []
    for category in ALLOWED_CATEGORIES:
        if category in words:
            found.append(category)

    if len(found) == 1:
        return "classified_keyword", found[0]

    return "flagged_invalid_output", None



# ---------------- Step 6: full pipeline, zero-shot vs few-shot ----------------
def run_pipeline(system_prompt, token_budget=None):
    """Run every ticket once, in file order. Returns one row per ticket."""
    global transient_errors

    mock_llm.reset()
    transient_errors = 0
    total_tokens = 0
    results = []

    for index, ticket in tickets.iterrows():
        status = ticket["guardrail_status"]
        category = None
        tokens_used = 0

        # Only tickets that passed the input guardrails go to the model
        if pd.isna(status):

            # Token budget check (used in step 7)
            if token_budget is not None and total_tokens >= token_budget:
                status = "budget_exhausted"
            else:
                response = call_llm(ticket["ticket_id"], ticket["redacted_text"], system_prompt)

                if response is None:
                    status = "flagged_invalid_output"     # no answer, send to a human
                else:
                    tokens_used = response.usage.total_tokens
                    total_tokens = total_tokens + tokens_used
                    status, category = validate_reply(response.choices[0].message.content)

        results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "status": status,
            "category": category,
            "true_category": ticket["true_category"],
            "tokens_used": tokens_used,
        })

    return pd.DataFrame(results)


def summarise_run(run):
    """Classified count, accuracy, total tokens and tokens per classified ticket."""
    classified = run[run["status"].isin(["classified_json", "classified_keyword"])]
    correct = (classified["category"] == classified["true_category"]).sum()
    total_tokens = run["tokens_used"].sum()

    return {
        "classified": len(classified),
        "correct": correct,
        "accuracy_pct": round(correct / len(classified) * 100, 1),
        "total_tokens": total_tokens,
        "tokens_per_classified": round(total_tokens / len(classified), 1),
    }


zero_shot_run = run_pipeline(ZERO_SHOT_PROMPT)
few_shot_run = run_pipeline(FEW_SHOT_PROMPT)

zero_shot_summary = summarise_run(zero_shot_run)
few_shot_summary = summarise_run(few_shot_run)

# Confusion table for the zero-shot run (classified tickets only)
zero_classified = zero_shot_run[zero_shot_run["status"].isin(["classified_json", "classified_keyword"])]
confusion = pd.crosstab(zero_classified["true_category"], zero_classified["category"])

# ---------------- Step 6 checks ----------------
print("Zero-shot:", zero_shot_summary)
print("Few-shot:", few_shot_summary)
print(confusion)

# ---------------- Step 7: budget demo, routing and output files ----------------
HUMAN_REVIEW_STATUSES = ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]

segments = pd.read_csv(OUTPUT_DIR / "customer_segments.csv")


def add_routing(run):
    """Add segment_name, priority and route to a pipeline run, using the Part B segments."""
    run = run.merge(segments[["customer_id", "segment_name", "is_premium"]], on="customer_id", how="left")

    # A customer with no row in the segment file is treated as not premium
    run["is_premium"] = run["is_premium"].fillna(False).astype(bool)

    routes = []
    priorities = []

    for index, row in run.iterrows():
        if row["status"] in HUMAN_REVIEW_STATUSES:
            route = "human_review"
            priority = "NORMAL"
        elif row["category"] == "fraud":
            route = "security_team"
            priority = "HIGH"
        elif row["is_premium"]:
            route = "premium_desk"
            priority = "HIGH"
        else:
            route = "support_queue"
            priority = "NORMAL"

        routes.append(route)
        priorities.append(priority)

    run["route"] = routes
    run["priority"] = priorities

    columns = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
    return run[columns]


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# (a) Budget demo: zero-shot prompt with a small budget
DEMO_BUDGET = 6000
budget_demo = add_routing(run_pipeline(ZERO_SHOT_PROMPT, token_budget=DEMO_BUDGET))
budget_demo.to_csv(OUTPUT_DIR / "triage_budget_demo.csv", index=False)

# (b) Final output: few-shot prompt with a large budget
FINAL_BUDGET = 100000
triage_results = add_routing(run_pipeline(FEW_SHOT_PROMPT, token_budget=FINAL_BUDGET))
triage_results.to_csv(OUTPUT_DIR / "triage_results.csv", index=False)

# ---------------- Step 7 checks ----------------
print("Budget demo - tickets budget_exhausted:", (budget_demo["status"] == "budget_exhausted").sum())
print("Budget demo - tokens used:", budget_demo["tokens_used"].sum())
print("Final rows:", len(triage_results))
print(triage_results["status"].value_counts())
print(pd.crosstab(triage_results["route"], triage_results["priority"]))