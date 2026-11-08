import pandas as pd
from pathlib import Path
import mock_llm
import json
import sqlite3
from datetime import datetime
import time

# ---------------- Step 1: chunk the policy documents ----------------
BASE_DIR = Path(__file__).resolve().parent
POLICY_DIR = BASE_DIR / "data" / "policies"
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"
STATE_DIR = BASE_DIR / "state"
DB_PATH = BASE_DIR / "data" / "instamart.db"

CHUNK_SIZE = 40
OVERLAP = 10


def chunk_words(text, chunk_size, overlap):
    """Split text into chunks of chunk_size words, each starting (chunk_size - overlap) words after the last."""
    words = text.split()
    step = chunk_size - overlap
    chunks = []

    start = 0
    while True:
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))

        # Stop once this chunk reaches or passes the last word
        if end >= len(words):
            break
        start = start + step

    return chunks


def build_chunks(chunk_size, overlap):
    """Chunk all five policy files and keep the source file name with each chunk."""
    all_chunks = []

    for file_path in sorted(POLICY_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")

        for chunk in chunk_words(text, chunk_size, overlap):
            all_chunks.append({"source": file_path.name, "text": chunk})

    return all_chunks


chunks = build_chunks(CHUNK_SIZE, OVERLAP)

# ---------------- Step 1 checks ----------------
chunk_df = pd.DataFrame(chunks)
print("Total chunks:", len(chunk_df))
print(chunk_df["source"].value_counts())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Step 2: vectorise and retrieve ----------------
def build_index(chunk_list):
    """Fit TF-IDF on the chunk texts and keep everything needed for retrieval."""
    texts = [chunk["text"] for chunk in chunk_list]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return {"chunks": chunk_list, "vectorizer": vectorizer, "matrix": matrix}


policy_index = build_index(chunks)


def retrieve(query, k=3, index=None):
    """Return the k chunks most similar to the query as [{source, score, text}, ...]."""
    if index is None:
        index = policy_index

    query_vector = index["vectorizer"].transform([query])
    scores = cosine_similarity(query_vector, index["matrix"])[0]

    # Chunk positions sorted from highest to lowest score
    ranked = scores.argsort()[::-1][:k]

    results = []
    for position in ranked:
        chunk = index["chunks"][position]
        results.append({
            "source": chunk["source"],
            "score": round(float(scores[position]), 4),
            "text": chunk["text"],
        })
    return results


top_results = retrieve("refund policy late delivery")

# ---------------- Step 2 checks ----------------
for result in top_results:
    print(result["source"], result["score"], result["text"][:60])

    # ---------------- Step 3: augment and answer ----------------
POLICY_SYSTEM_PROMPT = (
    "You are the Instamart POLICY ASSISTANT. Answer the question using only the policy context provided. "
    "If the answer is not in the context, say: I could not find that information in the policy documents."
)


def answer(question, index=None):
    """Retrieve three chunks, ask the model, and return its answer with the sources used."""
    results = retrieve(question, k=3, index=index)

    # Build the user message: CONTEXT, three [source] chunks, then QUESTION
    context_lines = []
    for result in results:
        context_lines.append(f"[{result['source']}]")
        context_lines.append(result["text"])
    context = "\n".join(context_lines)

    user_message = f"CONTEXT:\n{context}\nQUESTION:\n{question}"

    messages = [
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    response = mock_llm.chat(messages)

    sources = [result["source"] for result in results]
    return response.choices[0].message.content, sources


sample_answer, sample_sources = answer("Who must approve a refund above INR 1000?")
fallback_answer, fallback_sources = answer("Who won the cricket world cup final?")

# ---------------- Step 3 checks ----------------
print("Answer:", sample_answer)
print("Sources:", sample_sources)
print("Unanswerable:", fallback_answer)

# ---------------- Step 4: evaluate retrieval and grounding ----------------
policy_eval = pd.read_csv(RAW_DIR / "policy_eval.csv", keep_default_na=False)


def evaluate(index):
    """Run all 14 questions and return the summary counts plus one row per question."""
    rows = []

    for i, item in policy_eval.iterrows():
        reply, sources = answer(item["question"], index=index)
        answerable = item["expected_doc"] != "NONE"

        if answerable:
            doc_found = item["expected_doc"] in sources
            fact_found = item["expected_fact"] in reply      # case-sensitive substring
            fallback = None
        else:
            doc_found = None
            fact_found = None
            fallback = reply.startswith("I could not find")

        rows.append({
            "question": item["question"],
            "answerable": answerable,
            "doc_in_top3": doc_found,
            "fact_in_answer": fact_found,
            "fallback": fallback,
            "answer": reply,
        })

    detail = pd.DataFrame(rows)
    answerable_rows = detail[detail["answerable"]]
    unanswerable_rows = detail[~detail["answerable"]]

    summary = {
        "doc_in_top3": f"{int(answerable_rows['doc_in_top3'].sum())}/{len(answerable_rows)}",
        "fact_in_answer": f"{int(answerable_rows['fact_in_answer'].sum())}/{len(answerable_rows)}",
        "fallback": f"{int(unanswerable_rows['fallback'].sum())}/{len(unanswerable_rows)}",
    }
    return summary, detail


# Overlap 10 (main index)
summary_overlap10, detail_overlap10 = evaluate(policy_index)

# Overlap 0 (separate index, main index untouched)
chunks_overlap0 = build_chunks(CHUNK_SIZE, 0)
index_overlap0 = build_index(chunks_overlap0)
summary_overlap0, detail_overlap0 = evaluate(index_overlap0)

# Questions that fail with overlap 0
failed_overlap0 = detail_overlap0[
    detail_overlap0["answerable"]
    & ((detail_overlap0["doc_in_top3"] == False) | (detail_overlap0["fact_in_answer"] == False))
]

# ---------------- Step 4 checks ----------------
print("Overlap 10:", summary_overlap10)
print("Overlap 0 chunks:", len(chunks_overlap0))
print("Overlap 0:", summary_overlap0)
print(failed_overlap0[["question", "doc_in_top3", "fact_in_answer", "answer"]].to_string(index=False))

# ---------------- Step 5: tools and the tool map ----------------
LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"


def get_order(order_id):
    """Look up one order in the Part A database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
            (order_id,),
        )
        row = cursor.fetchone()

    if row is None:
        return {"error": "order not found"}

    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": row[2],
        "delivery_minutes": row[3],     # None for cancelled orders
    }


def search_policy(query):
    """Search the policy documents with the retrieval from step 2."""
    return {"results": retrieve(query)}


def calculate_refund(order_amount, delivery_minutes):
    """Full refund if delivered in more than 45 minutes, otherwise 0."""
    if delivery_minutes is not None and delivery_minutes > 45:
        refund = round(float(order_amount), 2)
    else:
        refund = 0
    return {"refund_inr": refund}


def write_ledger(tool_name, order_id, refund_inr):
    """Append one JSON line to the refund ledger."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": refund_inr,
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def issue_refund(order_id, refund_inr):
    """Issue a refund of INR 1000 or less and record it in the ledger."""
    write_ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}


def escalate_to_human(order_id, refund_inr):
    """Send a refund above INR 1000 to a human lead and record it in the ledger."""
    write_ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order in the database and return its status, amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Search the policy documents and return the three most relevant chunks.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate_refund",
        "description": "Work out the refund in INR from the order amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"},
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Issue a refund of INR 1000 or less and record it in the refund ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Send a refund above INR 1000 to a human support lead and record it in the refund ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


def execute_tool(name, arguments):
    """Look the tool up by name and call it with the model's arguments."""
    if name not in TOOL_MAP:
        return {"error": f"unknown tool {name}"}
    function = TOOL_MAP[name]
    return function(**arguments)


# ---------------- Step 5 checks ----------------
print([tool["name"] for tool in TOOLS_SPEC])
print(execute_tool("get_order", {"order_id": "ORD-100048"}))
print(execute_tool("get_order", {"order_id": "ORD-100025"}))
print(execute_tool("get_order", {"order_id": "ORD-999999"}))
print(execute_tool("calculate_refund", {"order_amount": 128.25, "delivery_minutes": 60}))
print(execute_tool("calculate_refund", {"order_amount": 128.25, "delivery_minutes": 30}))
print(execute_tool("search_policy", {"query": "refund policy late delivery"})["results"][0]["source"])

# ---------------- Step 6: the ReAct agent loop ----------------
AGENT_RUNS_PATH = OUTPUT_DIR / "agent_runs.jsonl"

AGENT_SYSTEM_PROMPT = (
    "You are the Instamart refund AGENT. Use the tools to look up the order, check the refund policy, "
    "calculate the refund, and then issue it or escalate it to a human. "
    "Finish with one line starting with REFUND, ESCALATE, DENIED or NOT_FOUND, followed by a colon."
)

# Start each script run with empty output files so lines don't pile up across runs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for path in [AGENT_RUNS_PATH, LEDGER_PATH]:
    if path.exists():
        path.unlink()


def handoff_summary(reason, steps):
    """Short summary for the human who takes over a stopped run."""
    tools_run = [step["tool"] for step in steps]
    return f"Handed to a human agent ({reason}). Tools completed: {tools_run if tools_run else 'none'}."


STATE_DIR.mkdir(parents=True, exist_ok=True)
for old_checkpoint in STATE_DIR.glob("*.json"):
    old_checkpoint.unlink()

run_counter = 0


class SimulatedCrash(Exception):
    """Raised on purpose to test resuming from a checkpoint."""


def save_checkpoint(state):
    """Write the current run state to state/<run_id>.json."""
    checkpoint = {
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "messages": state["messages"],
        "tokens_used": state["tokens_used"],
        "llm_calls": state["llm_calls"],
        "step": len(state["steps"]),
        "tools_executed": [step["tool"] for step in state["steps"]],
        "steps": state["steps"],
    }
    with open(STATE_DIR / f"{state['run_id']}.json", "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def agent_loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools=None):
    """The ReAct loop. Used by run_agent for a new run and by resume_agent after a crash."""
    start_time = time.monotonic()
    status = None
    final_answer = None

    while True:
        # (c) time limit, checked at the top of every iteration
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            status = "FALLBACK_TIMEOUT"
            final_answer = handoff_summary("time limit reached", state["steps"])
            break

        # (a) iteration limit
        if state["llm_calls"] >= max_iterations:
            status = "FALLBACK_MAX_ITERATIONS"
            final_answer = handoff_summary(f"no answer after {max_iterations} model calls", state["steps"])
            break

        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens

        # (b) token limit, checked right after each call, before running any tool
        if state["tokens_used"] > token_limit:
            status = "FALLBACK_TOKEN_LIMIT"
            final_answer = handoff_summary(f"token limit of {token_limit} exceeded", state["steps"])
            break

        message = response.choices[0].message

        # No tool call: this is the final answer
        if not message.tool_calls:
            final_answer = message.content
            status = final_answer.split(":")[0]
            break

        # Run the requested tool and add both messages to the history
        tool_call = message.tool_calls[0]
        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tool_call.id, "name": tool_call.name, "arguments": tool_call.arguments}],
        })

        observation = execute_tool(tool_call.name, tool_call.arguments)

        state["messages"].append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.name,
            "content": json.dumps(observation),
        })

        state["steps"].append({
            "step": len(state["steps"]) + 1,
            "tool": tool_call.name,
            "arguments": tool_call.arguments,
            "observation": observation,
        })

        # Checkpoint after every tool observation
        save_checkpoint(state)

        # Simulated crash for the resume demo (only when asked for)
        if crash_after_tools is not None and len(state["steps"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool {crash_after_tools}")

    return {
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "status": status,
        "final_answer": final_answer,
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
    }


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              crash_after_tools=None):
    global run_counter
    run_counter += 1

    state = {
        "run_id": f"run_{run_counter:03d}_{scenario_id}",
        "scenario_id": scenario_id,
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "steps": [],
        "llm_calls": 0,
        "tokens_used": 0,
    }
    return agent_loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools)


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    """Load a checkpoint from state/ and continue the run from where it stopped."""
    with open(STATE_DIR / f"{run_id}.json", encoding="utf-8") as f:
        checkpoint = json.load(f)

    state = {
        "run_id": checkpoint["run_id"],
        "scenario_id": checkpoint["scenario_id"],
        "messages": checkpoint["messages"],
        "steps": checkpoint["steps"],
        "llm_calls": checkpoint["llm_calls"],
        "tokens_used": checkpoint["tokens_used"],
    }
    return agent_loop(state, max_iterations, token_limit, time_limit_seconds)

def refund_amount(result):
    """Refund issued or escalated in a run, otherwise 0."""
    for step in result["steps"]:
        if step["tool"] in ["issue_refund", "escalate_to_human"]:
            return step["observation"]["refund_inr"]
    return 0


def log_run(result):
    """Append one JSON line per run to outputs/agent_runs.jsonl."""
    with open(AGENT_RUNS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


scenarios = pd.read_csv(RAW_DIR / "agent_scenarios.csv", keep_default_na=False)
scenario_results = []

for i, scenario in scenarios.iterrows():
    result = run_agent(scenario["message"], scenario["scenario_id"])
    log_run(result)
    scenario_results.append({
        "scenario_id": result["scenario_id"],
        "status": result["status"],
        "refund_inr": refund_amount(result),
        "expected_outcome": scenario["expected_outcome"],
        "expected_refund_inr": float(scenario["expected_refund_inr"]),
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"],
    })

scenario_table = pd.DataFrame(scenario_results)

# ---------------- Step 6 checks ----------------
print(scenario_table.to_string(index=False))
print("Lines in agent_runs.jsonl:", len(AGENT_RUNS_PATH.read_text(encoding="utf-8").splitlines()))
print(LEDGER_PATH.read_text(encoding="utf-8"))

# ---------------- Step 7: stop-safely demonstrations ----------------
s1_message = scenarios.loc[scenarios["scenario_id"] == "S1", "message"].iloc[0]

token_demo = run_agent(s1_message, "S1", token_limit=200)
timeout_demo = run_agent(s1_message, "S1", time_limit_seconds=0)

# ---------------- Step 7 checks ----------------
for name, demo in [("Token limit 200", token_demo), ("Time limit 0", timeout_demo)]:
    print(name, "|", demo["status"], "| model calls:", demo["llm_calls"],
          "| tokens:", demo["tokens_used"], "| tools run:", [step["tool"] for step in demo["steps"]])
    print("   ", demo["final_answer"])
print("Ledger lines (should still be 3):", len(LEDGER_PATH.read_text(encoding="utf-8").splitlines()))

# ---------------- Step 8: crash and resume ----------------
crash_run_id = None
tools_before_crash = []

try:
    run_agent(s1_message, "S1", crash_after_tools=2)
except SimulatedCrash as error:
    crash_message = str(error)
    crash_run_id = f"run_{run_counter:03d}_S1"
    with open(STATE_DIR / f"{crash_run_id}.json", encoding="utf-8") as f:
        tools_before_crash = json.load(f)["tools_executed"]

resumed = resume_agent(crash_run_id)
tools_after_resume = [step["tool"] for step in resumed["steps"]][len(tools_before_crash):]

# ---------------- Step 8 checks ----------------
print("Crash:", crash_message, "| run id:", crash_run_id)
print("Tools before crash:", tools_before_crash)
print("Tools after resume:", tools_after_resume)
print("All tools in order:", [step["tool"] for step in resumed["steps"]])
print("Final:", resumed["status"], refund_amount(resumed), "| model calls:", resumed["llm_calls"])
print("Checkpoint files:", sorted(p.name for p in STATE_DIR.glob("*.json")))