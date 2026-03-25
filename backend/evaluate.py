"""
Mini RAG - Quality Analysis / Evaluation Script
Indecimal AI Assistant

Runs 12 test questions against the RAG pipeline and evaluates:
- Relevance of retrieved chunks
- Groundedness (hallucination proxy)
- Completeness of answers
"""

import json
import urllib.request

API_URL = "http://localhost:8000/query"

TEST_QUESTIONS = [
    "What does Indecimal promise to customers?",
    "What are the stages of the Indecimal customer journey?",
    "How does Indecimal ensure transparent pricing?",
    "What are the package prices per sqft?",
    "What steel brand is used in the Pinnacle package?",
    "What cement is used in the Infinia package?",
    "What is the main door wallet amount for the Premier package?",
    "What flooring allowance does the Essential package offer for living and dining?",
    "What painting brand is used for exterior in the Pinnacle package?",
    "How does Indecimal protect customer payments?",
    "How many quality checkpoints does Indecimal have?",
    "What does the zero cost maintenance program cover?",
]

EXPECTED_KEYWORDS = {
    "What does Indecimal promise to customers?":                           ["confidence", "transparent", "quality", "warranty"],
    "What are the stages of the Indecimal customer journey?":              ["request", "design", "financing", "handover"],
    "How does Indecimal ensure transparent pricing?":                      ["transparent", "hidden", "pricing", "plans"],
    "What are the package prices per sqft?":                               ["1,851", "1,995", "2,250", "2,450"],
    "What steel brand is used in the Pinnacle package?":                   ["TATA", "80,000"],
    "What cement is used in the Infinia package?":                         ["Birla", "Ramco", "390"],
    "What is the main door wallet amount for the Premier package?":        ["30,000", "teak"],
    "What flooring allowance does the Essential package offer for living and dining?": ["50", "tiles"],
    "What painting brand is used for exterior in the Pinnacle package?":   ["Apex Ultima", "Asian Paints"],
    "How does Indecimal protect customer payments?":                       ["escrow", "verified", "disbursed"],
    "How many quality checkpoints does Indecimal have?":                   ["445"],
    "What does the zero cost maintenance program cover?":                  ["plumbing", "electrical", "roofing", "painting"],
}


def call_api(question):
    payload = json.dumps({"question": question}).encode()
    req = urllib.request.Request(
        API_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def score_answer(question, answer, chunks):
    answer_lower = answer.lower()
    keywords = EXPECTED_KEYWORDS.get(question, [])
    matched  = [kw for kw in keywords if kw.lower() in answer_lower]
    relevance = len(matched) / len(keywords) if keywords else 1.0

    chunk_words = set()
    for c in chunks:
        chunk_words.update(c["text"].lower().split())
    answer_words = set(answer_lower.split())
    overlap      = len(answer_words & chunk_words) / max(len(answer_words), 1)
    groundedness = min(overlap * 2, 1.0)
    completeness = min(len(answer) / 200, 1.0)

    return {
        "relevance":    round(relevance, 2),
        "groundedness": round(groundedness, 2),
        "completeness": round(completeness, 2),
        "keywords_hit": matched,
    }


def run_evaluation():
    print("=" * 70)
    print("  INDECIMAL RAG — EVALUATION REPORT")
    print("=" * 70)

    results = []
    total_rel = total_gnd = total_cmp = 0.0

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i:02d}/{len(TEST_QUESTIONS)}] {question}")
        print("-" * 60)
        try:
            resp   = call_api(question)
            answer = resp.get("answer", "")
            chunks = resp.get("retrieved_chunks", [])
            scores = score_answer(question, answer, chunks)

            print(f"  Answer (first 200 chars): {answer[:200]}...")
            print(f"  Retrieved chunks: {len(chunks)}")
            for c in chunks:
                print(f"    [{c['source']}] score={c['score']:.3f} | {c['text'][:80]}...")
            print(f"  Relevance: {scores['relevance']:.0%}  Groundedness: {scores['groundedness']:.0%}  Completeness: {scores['completeness']:.0%}")
            print(f"  Keywords matched: {scores['keywords_hit']}")

            total_rel += scores["relevance"]
            total_gnd += scores["groundedness"]
            total_cmp += scores["completeness"]
            results.append({"question": question, "scores": scores, "ok": True})

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"question": question, "error": str(e), "ok": False})

    n = len(TEST_QUESTIONS)
    print("\n" + "=" * 70)
    print("  AGGREGATE SCORES")
    print("=" * 70)
    print(f"  Average Relevance:    {total_rel/n:.0%}")
    print(f"  Average Groundedness: {total_gnd/n:.0%}")
    print(f"  Average Completeness: {total_cmp/n:.0%}")
    print(f"  Success Rate:         {sum(r['ok'] for r in results)}/{n}")

    with open("evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Report saved to evaluation_report.json")


if __name__ == "__main__":
    run_evaluation()
