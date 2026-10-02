"""
LangGraph Agent Pipeline for Fact-Checker & Daily News Digest
Nodes: Claim Extractor → Evidence Retriever (ReAct) → Verdict Judge → Critic → Digest Writer
"""

import json
import os
import re
from datetime import datetime, date
from typing import Annotated, Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from tavily import TavilyClient

from rag_knowledge_base import get_rag

load_dotenv()

import logging
for _lg in ("google_genai", "google_genai.types", "vertexai"):
    logging.getLogger(_lg).setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────
# LLM & Clients — Gemini 3.1 Flash Lite only (fast, CPU-friendly)
# ─────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-3.1-flash-lite"


def _make_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.1,
        max_output_tokens=2048,
        timeout=45,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


class _GeminiLLM:
    """Gemini 3.1 Flash Lite with a short retry on transient errors (same model only)."""

    model_name = GEMINI_MODEL

    def invoke(self, messages):
        last_err = None
        for attempt in range(3):
            try:
                return _make_llm().invoke(messages)
            except Exception as e:
                last_err = e
                msg = str(e)
                transient = any(k in msg for k in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
                                                   "timeout", "Timeout", "deadline", "500"))
                if not transient:
                    raise
                print(f"[LLM] Transient error on {GEMINI_MODEL} (attempt {attempt + 1}/3): {msg[:100]}")
        raise last_err


llm = _GeminiLLM()


def _text(response) -> str:
    """Normalize LLM response content (str or list of blocks) to plain text."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(getattr(block, "text", "") or "")
        return "".join(parts).strip()
    return str(content).strip()

_tavily_key = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=_tavily_key) if _tavily_key and "your_" not in _tavily_key else None


# ─────────────────────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────────────────────
class ClaimResult(TypedDict):
    claim: str
    verdict: str                      # True | False | Misleading | Unverifiable
    confidence: float                  # 0.0 – 1.0
    evidence_for: list[str]
    evidence_against: list[str]
    sources: list[dict]               # [{url, title, credibility}]
    summary: str
    critique: Optional[str]
    iterations: int


class AgentState(TypedDict):
    # Input
    user_input: str
    mode: str                          # "fact_check" | "digest"
    topic: Optional[str]

    # Pipeline data
    claims: list[str]                  # Extracted individual claims
    claim_results: list[ClaimResult]   # Final verdicts
    current_claim_index: int
    current_claim_result: Optional[ClaimResult]

    # Reflection
    critic_feedback: Optional[str]
    reflection_rounds: int
    max_reflection_rounds: int

    # Output
    digest_markdown: str
    digest_saved: bool
    export_path: str

    # Messages for sub-LLM calls
    messages: Annotated[list, add_messages]


# ─────────────────────────────────────────────────────────────
# Helper: Tavily search
# ─────────────────────────────────────────────────────────────
def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Search Tavily and return structured results."""
    if tavily is None:
        print("[Tavily] No valid TAVILY_API_KEY — skipping live search.")
        return []
    try:
        results = tavily.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_raw_content=False
        )
        hits = []
        for r in results.get("results", []):
            hits.append({
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0)
            })
        return hits
    except Exception as e:
        print(f"[Tavily] Search error: {e}")
        return []


def get_credibility(url: str) -> int:
    rag = get_rag()
    info = rag.get_source_credibility(url)
    return info.get("credibility", 5)


# ─────────────────────────────────────────────────────────────
# NODE 1 — Claim Extractor
# ─────────────────────────────────────────────────────────────
def claim_extractor_node(state: AgentState) -> AgentState:
    """Splits user input into individual, checkable factual claims."""
    user_input = state["user_input"]
    mode = state.get("mode", "fact_check")

    if mode == "digest":
        # Ground digest claims in real, live news results (no invented stories)
        topic = state.get("topic", "Technology")
        headlines = tavily_search(f"{topic} news {date.today()}", max_results=8)
        news_block = "\n".join(
            f"- {h['title']} ({h['url']}): {h['content'][:250]}" for h in headlines
        ) or "- No recent articles found."
        prompt = f"""You are a news research assistant. Build a daily verified news digest on "{topic}" ({date.today()}).

Real recent articles:
{news_block}

Generate exactly 5 checkable factual claims drawn ONLY from the articles above.
Each claim must be a single specific statement a reader could verify.

Return ONLY a JSON array of strings.
Example: ["Company X released product Y on date Z", ...]"""
    else:
        prompt = f"""You are a fact-checking assistant. Extract all individual, checkable factual claims from the following text.

Rules:
- Each claim must be a single, specific, verifiable statement
- Ignore opinions, predictions, or questions
- If the text IS a single claim, return it as-is
- Do not paraphrase — keep the original wording where possible

Return ONLY a JSON array of strings.
Example: ["The earth is flat", "Vitamin C cures the common cold"]

Text to analyze:
{user_input}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = _text(response)

    # Parse JSON array
    try:
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
        claims = json.loads(raw)
        if not isinstance(claims, list):
            claims = [user_input]
    except Exception:
        # Fallback: treat whole input as one claim
        claims = [user_input] if mode == "fact_check" else [f"Latest {state.get('topic', 'news')} developments"]

    if mode == "digest":
        claims = [str(c) for c in claims][:5]

    print(f"[ClaimExtractor] Extracted {len(claims)} claims.")
    return {
        **state,
        "claims": claims,
        "current_claim_index": 0,
        "claim_results": [],
        "reflection_rounds": 0,
        "max_reflection_rounds": state.get("max_reflection_rounds") or 2
    }


# ─────────────────────────────────────────────────────────────
# NODE 2 — Evidence Retriever (ReAct loop)
# ─────────────────────────────────────────────────────────────
def evidence_retriever_node(state: AgentState) -> AgentState:
    """Searches Tavily + RAG archive for evidence on the current claim."""
    claims = state["claims"]
    idx = state["current_claim_index"]

    if idx >= len(claims):
        return state

    claim = claims[idx]
    critic_feedback = state.get("critic_feedback")

    print(f"[EvidenceRetriever] Gathering evidence for claim {idx+1}/{len(claims)}: '{claim[:80]}...'")

    # 1. Search RAG archive first
    rag = get_rag()
    rag_hits = rag.search_similar_claims(claim, n_results=3)

    # 2+3. Tavily: run both queries in parallel (supporting + contradicting)
    support_query = f'fact check: "{claim}"'
    if critic_feedback:
        support_query += f" {critic_feedback}"
    counter_query = f'debunking myth: "{claim}" evidence against'

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_support = pool.submit(tavily_search, support_query, 4)
        fut_counter = pool.submit(tavily_search, counter_query, 3)
        support_results = fut_support.result()
        counter_results = fut_counter.result()

    # Combine and deduplicate sources
    all_sources = {}
    for r in support_results + counter_results:
        url = r["url"]
        if url and url not in all_sources:
            all_sources[url] = {
                "url": url,
                "title": r["title"],
                "content": r["content"][:500],
                "credibility": get_credibility(url)
            }

    # Build evidence strings
    evidence_for = [
        f"[{r['title']}] ({r['url']}): {r['content'][:300]}"
        for r in support_results if r.get("content")
    ]
    evidence_against = [
        f"[{r['title']}] ({r['url']}): {r['content'][:300]}"
        for r in counter_results if r.get("content")
    ]

    # Add RAG hits to evidence
    for hit in rag_hits:
        rag_text = f"[ARCHIVE] Similar claim: '{hit['claim']}' — Verdict: {hit['verdict']} (confidence {hit['confidence']}). {hit['summary']}"
        evidence_for.append(rag_text)

    return {
        **state,
        "current_claim_result": {
            "claim": claim,
            "verdict": "Unverifiable",
            "confidence": 0.0,
            "evidence_for": evidence_for,
            "evidence_against": evidence_against,
            "sources": list(all_sources.values()),
            "summary": "",
            "critique": None,
            "iterations": state.get("reflection_rounds", 0)
        },
        "critic_feedback": None
    }


# ─────────────────────────────────────────────────────────────
# NODE 3 — Verdict Judge
# ─────────────────────────────────────────────────────────────
def verdict_judge_node(state: AgentState) -> AgentState:
    """Weighs evidence and assigns a verdict with confidence and summary."""
    result = state.get("current_claim_result", {})
    if not result:
        return state

    claim = result["claim"]
    evidence_for = result["evidence_for"]
    evidence_against = result["evidence_against"]
    sources = result["sources"]

    prompt = f"""You are an expert fact-checker and evidence analyst. Your task is to weigh evidence and deliver a reasoned verdict.

CLAIM: {claim}

SUPPORTING EVIDENCE:
{chr(10).join(f'- {e}' for e in evidence_for[:6]) if evidence_for else '- None found'}

CONTRADICTING EVIDENCE:
{chr(10).join(f'- {e}' for e in evidence_against[:4]) if evidence_against else '- None found'}

SOURCES (with credibility 1-10):
{chr(10).join(f'- [{s["credibility"]}/10] {s["url"]}: {s["title"]}' for s in sources[:8])}

Instructions:
1. Weigh evidence from BOTH sides objectively
2. Prioritize high-credibility sources (8-10)  
3. Choose ONE verdict: True | False | Misleading | Unverifiable
   - True: claim is accurate based on strong evidence
   - False: claim is demonstrably incorrect
   - Misleading: claim has some truth but is presented in a way that creates a false impression
   - Unverifiable: insufficient reliable evidence exists — do NOT guess
4. Assign confidence (0.0–1.0): high confidence only with multiple high-credibility sources
5. Write a balanced, objective 3-4 sentence summary that:
   - States what the evidence shows
   - Acknowledges counter-evidence if any
   - Cites specific sources
6. AVOID political bias; MUST present both sides

Return ONLY valid JSON with these exact keys:
{{
  "verdict": "True|False|Misleading|Unverifiable",
  "confidence": 0.85,
  "summary": "Balanced summary here...",
  "key_evidence_for": ["Brief point 1", "Brief point 2"],
  "key_evidence_against": ["Brief point 1"]
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = _text(response)

    try:
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
        data = json.loads(raw)
    except Exception:
        data = {
            "verdict": "Unverifiable",
            "confidence": 0.0,
            "summary": "Could not parse verdict. Evidence was inconclusive.",
            "key_evidence_for": [],
            "key_evidence_against": []
        }

    updated_result = {
        **result,
        "verdict": data.get("verdict", "Unverifiable"),
        "confidence": data.get("confidence", 0.0),
        "summary": data.get("summary", ""),
        "evidence_for": data.get("key_evidence_for", result["evidence_for"][:3]),
        "evidence_against": data.get("key_evidence_against", result["evidence_against"][:3]),
    }

    print(f"[VerdictJudge] Verdict: {updated_result['verdict']} (confidence: {updated_result['confidence']:.0%})")

    return {**state, "current_claim_result": updated_result}


# ─────────────────────────────────────────────────────────────
# NODE 4 — Critic (Reflection)
# ─────────────────────────────────────────────────────────────
def critic_node(state: AgentState) -> AgentState:
    """Reviews verdict for weak evidence or bias; sends back if needed."""
    result = state.get("current_claim_result", {})
    reflection_rounds = state.get("reflection_rounds", 0)
    max_rounds = state.get("max_reflection_rounds", 2)

    if not result:
        return state

    prompt = f"""You are a senior editorial fact-checking critic. Review this fact-check result for quality.

CLAIM: {result['claim']}
VERDICT: {result['verdict']}
CONFIDENCE: {result['confidence']:.0%}
SUMMARY: {result['summary']}
EVIDENCE FOR: {json.dumps(result['evidence_for'][:3], indent=2)}
EVIDENCE AGAINST: {json.dumps(result['evidence_against'][:3], indent=2)}
SOURCES COUNT: {len(result['sources'])}
REVIEW ROUND: {reflection_rounds + 1}/{max_rounds}

Evaluate:
1. Is there sufficient evidence from credible sources?
2. Is the verdict well-supported?
3. Is there potential political or editorial bias?
4. Are both sides represented fairly?
5. Is confidence level appropriate for the evidence?

Return JSON with:
{{
  "approved": true|false,
  "issues": ["Issue 1 if any", "Issue 2 if any"],
  "search_suggestions": ["Better search query 1", "Query 2"],
  "adjusted_confidence": 0.85,
  "critic_notes": "Brief notes for the digest"
}}

Approve (true) by DEFAULT — only reject when a specific, critical flaw exists: inflated confidence, clear one-sided bias, or a key high-credibility source was missed. Rejection costs another full search round, so do not reject for minor issues."""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = _text(response)

    try:
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
        critique = json.loads(raw)
    except Exception:
        critique = {"approved": True, "issues": [], "search_suggestions": [], "adjusted_confidence": result["confidence"], "critic_notes": ""}

    approved = critique.get("approved", True)
    try:
        adjusted_conf = float(critique.get("adjusted_confidence", result["confidence"]))
    except (TypeError, ValueError):
        adjusted_conf = result["confidence"]

    # Sanity clamp: critic may only nudge confidence (±0.15), keep verdict/confidence consistent
    original_conf = float(result.get("confidence", 0.5) or 0.5)
    adjusted_conf = max(original_conf - 0.15, min(original_conf + 0.15, adjusted_conf))
    adjusted_conf = max(0.0, min(0.99, adjusted_conf))
    if result.get("verdict") in ("True", "False") and adjusted_conf < 0.5:
        adjusted_conf = 0.5

    # Update confidence based on critic
    updated_result = {**result, "confidence": adjusted_conf, "critique": critique.get("critic_notes", "")}

    suggestions = [s for s in critique.get("search_suggestions", []) if str(s).strip()]
    if not approved and reflection_rounds < max_rounds and suggestions:
        feedback = ". ".join(suggestions)
        print(f"[Critic] Rejected — requesting more evidence. Round {reflection_rounds+1}/{max_rounds}")
        return {
            **state,
            "current_claim_result": updated_result,
            "critic_feedback": feedback,
            "reflection_rounds": reflection_rounds + 1
        }
    else:
        print(f"[Critic] Approved verdict: {result['verdict']}")
        # Finalize this claim
        claim_results = state.get("claim_results", [])
        claim_results = claim_results + [updated_result]
        return {
            **state,
            "current_claim_result": updated_result,
            "claim_results": claim_results,
            "current_claim_index": state["current_claim_index"] + 1,
            "reflection_rounds": 0,
            "critic_feedback": None
        }


# ─────────────────────────────────────────────────────────────
# NODE 5 — Digest Writer
# ─────────────────────────────────────────────────────────────
def digest_writer_node(state: AgentState) -> AgentState:
    """Compiles all verdicts into a structured Markdown digest."""
    claim_results = state.get("claim_results", [])
    mode = state.get("mode", "fact_check")
    topic = state.get("topic", "General")

    if not claim_results:
        return {**state, "digest_markdown": "# No claims to report.", "digest_saved": False}

    # Build verdict summary
    verdict_counts = {}
    for r in claim_results:
        v = r.get("verdict", "Unverifiable")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    # Ask LLM to write a compelling summary
    claims_json = json.dumps([{
        "claim": r["claim"],
        "verdict": r["verdict"],
        "confidence": r["confidence"],
        "summary": r["summary"]
    } for r in claim_results], indent=2)

    summary_prompt = f"""Write a concise executive summary (3-4 sentences) for a fact-check/news digest report.
Mode: {mode}
Topic: {topic}
Date: {date.today()}
Claims checked: {len(claim_results)}
Verdict distribution: {verdict_counts}

Claims:
{claims_json}

Keep it objective and journalistic. No markdown, plain text only."""

    summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
    exec_summary = _text(summary_response)

    # Build Markdown
    verdict_emoji = {
        "True": "✅",
        "False": "❌",
        "Misleading": "⚠️",
        "Unverifiable": "❓"
    }

    if mode == "digest":
        title = f"📰 Daily News Digest — {topic} ({date.today()})"
    else:
        title = f"🔍 Fact-Check Report ({date.today()})"

    md_lines = [
        f"## {title}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Claims Checked:** {len(claim_results)}  ",
        f"**Verdicts:** " + " | ".join(f"{v}: {c}" for v, c in verdict_counts.items()),
        "",
        "---",
        "",
        "### Executive Summary",
        "",
        exec_summary,
        "",
        "---",
        "",
        "### Detailed Findings",
        ""
    ]

    for i, r in enumerate(claim_results, 1):
        verdict = r.get("verdict", "Unverifiable")
        emoji = verdict_emoji.get(verdict, "❓")
        confidence_pct = int(r.get("confidence", 0) * 100)

        md_lines += [
            f"#### {i}. {emoji} {verdict.upper()} — Confidence: {confidence_pct}%",
            "",
            f"**Claim:** *\"{r['claim']}\"*",
            "",
            f"**Summary:** {r.get('summary', 'No summary available.')}",
            ""
        ]

        # Evidence
        ev_for = r.get("evidence_for", [])
        ev_against = r.get("evidence_against", [])

        if ev_for:
            md_lines += ["**Evidence Supporting:**"]
            for e in ev_for[:3]:
                md_lines.append(f"- {str(e)[:200]}")
            md_lines.append("")

        if ev_against:
            md_lines += ["**Evidence Against / Counter-points:**"]
            for e in ev_against[:3]:
                md_lines.append(f"- {str(e)[:200]}")
            md_lines.append("")

        # Sources
        sources = r.get("sources", [])
        if sources:
            md_lines += ["**Sources:**"]
            for s in sources[:5]:
                cred = s.get("credibility", 5)
                stars = "⭐" * min(int(cred / 2), 5)
                md_lines.append(f"- [{s.get('title', s.get('url', 'Source'))}]({s.get('url', '#')}) {stars}")
            md_lines.append("")

        if r.get("critique"):
            md_lines += [f"*Editor's Note: {r['critique']}*", ""]

        md_lines += ["---", ""]

    # Guardrail footer
    md_lines += [
        "### ⚖️ Methodology & Guardrails",
        "",
        "- Claims labeled **Unverifiable** where evidence was insufficient",
        "- Evidence presented from **both supporting and opposing** perspectives",
        "- Sources prioritized by credibility rating (fact-checkers, scientific journals, wire services)",
        "- **Political bias guardrail** applied: no partisan framing",
        "- Each verdict reviewed by an AI critic before publication",
        "",
        f"*This report was generated by the AI Fact-Checker pipeline on {datetime.now().strftime('%Y-%m-%d %H:%M')}.*"
    ]

    digest_markdown = "\n".join(md_lines)

    # Archive new fact-checks in RAG
    rag = get_rag()
    for r in claim_results:
        if r.get("verdict") and r.get("verdict") != "Unverifiable":
            try:
                rag.add_fact_check(
                    claim=r["claim"],
                    verdict=r["verdict"],
                    summary=r.get("summary", ""),
                    sources=[s.get("url", "") for s in r.get("sources", [])],
                    confidence=r.get("confidence", 0.5),
                    topic=topic
                )
            except Exception:
                pass

    # Save digest via MCP (save_digest tool: date + topic + claims)
    digest_saved = False
    try:
        import mcp_client
        mcp_payload = [{
            "claim": r.get("claim", ""),
            "verdict": r.get("verdict", "Unverifiable"),
            "confidence": r.get("confidence", 0.0),
            "summary": r.get("summary", ""),
            "sources": [s.get("url", "") for s in r.get("sources", []) if s.get("url")]
        } for r in claim_results]
        mcp_client.save_digest(
            date=str(date.today()),
            topic=topic if mode == "digest" else "Fact-Check",
            claims=mcp_payload,
            digest_summary=exec_summary
        )
        digest_saved = True
    except Exception as e:
        print(f"[DigestWriter] MCP save_digest failed: {e}")

    print(f"[DigestWriter] Compiled digest with {len(claim_results)} claims. Saved via MCP: {digest_saved}")

    return {
        **state,
        "digest_markdown": digest_markdown,
        "digest_saved": digest_saved,
        "export_path": ""
    }


# ─────────────────────────────────────────────────────────────
# Routing Functions
# ─────────────────────────────────────────────────────────────
def route_after_critic(state: AgentState) -> Literal["evidence_retriever", "check_more_claims", "digest_writer"]:
    """Route: if critic rejected → more evidence; if more claims → next claim; else digest."""
    if state.get("critic_feedback"):
        return "evidence_retriever"

    idx = state.get("current_claim_index", 0)
    claims = state.get("claims", [])

    if idx < len(claims):
        return "check_more_claims"
    else:
        return "digest_writer"


def route_more_claims(state: AgentState) -> Literal["evidence_retriever", "digest_writer"]:
    """Route to next claim or digest writing."""
    idx = state.get("current_claim_index", 0)
    claims = state.get("claims", [])
    if idx < len(claims):
        return "evidence_retriever"
    return "digest_writer"


# ─────────────────────────────────────────────────────────────
# Build LangGraph
# ─────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("claim_extractor", claim_extractor_node)
    graph.add_node("evidence_retriever", evidence_retriever_node)
    graph.add_node("verdict_judge", verdict_judge_node)
    graph.add_node("critic", critic_node)
    graph.add_node("digest_writer", digest_writer_node)
    graph.add_node("check_more_claims", lambda s: s)  # Pass-through router node

    # Edges
    graph.add_edge(START, "claim_extractor")
    graph.add_edge("claim_extractor", "evidence_retriever")
    graph.add_edge("evidence_retriever", "verdict_judge")
    graph.add_edge("verdict_judge", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "evidence_retriever": "evidence_retriever",
            "check_more_claims": "check_more_claims",
            "digest_writer": "digest_writer"
        }
    )

    graph.add_conditional_edges(
        "check_more_claims",
        route_more_claims,
        {
            "evidence_retriever": "evidence_retriever",
            "digest_writer": "digest_writer"
        }
    )

    graph.add_edge("digest_writer", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def run_fact_check(user_input: str) -> dict:
    """Run the fact-checking pipeline on arbitrary text/claim."""
    graph = build_graph()
    initial_state = {
        "user_input": user_input,
        "mode": "fact_check",
        "topic": None,
        "claims": [],
        "claim_results": [],
        "current_claim_index": 0,
        "current_claim_result": None,
        "critic_feedback": None,
        "reflection_rounds": 0,
        "max_reflection_rounds": 2,
        "digest_markdown": "",
        "digest_saved": False,
        "export_path": "",
        "messages": []
    }
    final_state = graph.invoke(initial_state)
    return final_state


def run_daily_digest(topic: str = "Technology") -> dict:
    """Generate a daily verified news digest on a topic."""
    graph = build_graph()
    initial_state = {
        "user_input": f"Give me today's verified {topic} news digest.",
        "mode": "digest",
        "topic": topic,
        "claims": [],
        "claim_results": [],
        "current_claim_index": 0,
        "current_claim_result": None,
        "critic_feedback": None,
        "reflection_rounds": 0,
        "max_reflection_rounds": 1,
        "digest_markdown": "",
        "digest_saved": False,
        "export_path": "",
        "messages": []
    }
    final_state = graph.invoke(initial_state)
    return final_state
