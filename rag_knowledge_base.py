"""
RAG Knowledge Base for Fact-Checker
Uses ChromaDB for vector storage of archived fact-checks and trusted sources.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"
TRUSTED_SOURCES_FILE = Path(__file__).parent / "trusted_sources.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


TRUSTED_SOURCES = {
    "snopes.com": {"credibility": 9, "type": "fact-checker", "notes": "Established fact-checking site"},
    "factcheck.org": {"credibility": 9, "type": "fact-checker", "notes": "Non-partisan fact-checker"},
    "politifact.com": {"credibility": 8, "type": "fact-checker", "notes": "Pulitzer Prize–winning fact-checker"},
    "reuters.com": {"credibility": 9, "type": "news", "notes": "International news agency, minimal bias"},
    "apnews.com": {"credibility": 9, "type": "news", "notes": "Associated Press – highly reliable"},
    "bbc.com": {"credibility": 8, "type": "news", "notes": "British public broadcaster"},
    "npr.org": {"credibility": 8, "type": "news", "notes": "National Public Radio, USA"},
    "who.int": {"credibility": 10, "type": "health", "notes": "World Health Organization"},
    "cdc.gov": {"credibility": 10, "type": "health", "notes": "Centers for Disease Control"},
    "nih.gov": {"credibility": 10, "type": "health", "notes": "National Institutes of Health"},
    "nature.com": {"credibility": 10, "type": "science", "notes": "Peer-reviewed scientific journal"},
    "science.org": {"credibility": 10, "type": "science", "notes": "AAAS Science journal"},
    "fullfact.org": {"credibility": 9, "type": "fact-checker", "notes": "UK's independent fact-checker"},
    "usatoday.com": {"credibility": 7, "type": "news", "notes": "USA Today – mainstream news"},
    "washingtonpost.com": {"credibility": 8, "type": "news", "notes": "Washington Post fact-checker"},
    "theguardian.com": {"credibility": 8, "type": "news", "notes": "The Guardian – reputable British paper"},
    "economist.com": {"credibility": 9, "type": "news", "notes": "The Economist – analytical journalism"},
    "afp.com": {"credibility": 9, "type": "news", "notes": "Agence France-Presse"},
    "checkyourfact.com": {"credibility": 8, "type": "fact-checker", "notes": "Daily Caller News Foundation"},
    "leadstories.com": {"credibility": 8, "type": "fact-checker", "notes": "Fact-checking hoaxes and rumors"},
}

# Seed fact-checks for RAG archive
SEED_FACT_CHECKS = [
    {
        "claim": "Drinking hot water cures viral infections",
        "verdict": "False",
        "summary": "Hot water does not cure viral infections. While warm liquids can soothe symptoms, no scientific evidence supports a curative effect. WHO and CDC recommend standard treatments.",
        "sources": ["who.int", "cdc.gov", "nih.gov"],
        "confidence": 0.95,
        "topic": "Health",
        "date": "2024-01-01"
    },
    {
        "claim": "Vaccines cause autism",
        "verdict": "False",
        "summary": "Extensive research involving millions of children has found no link between vaccines and autism. The original 1998 study was retracted due to fraud.",
        "sources": ["who.int", "cdc.gov", "nature.com"],
        "confidence": 0.99,
        "topic": "Health",
        "date": "2024-01-01"
    },
    {
        "claim": "5G technology spreads COVID-19",
        "verdict": "False",
        "summary": "5G is radio waves and cannot carry or transmit viruses. COVID-19 spreads through respiratory droplets. Many countries without 5G also had COVID outbreaks.",
        "sources": ["who.int", "fullfact.org", "reuters.com"],
        "confidence": 0.99,
        "topic": "Technology",
        "date": "2024-01-01"
    },
    {
        "claim": "The Great Wall of China is visible from space",
        "verdict": "False",
        "summary": "NASA astronauts have confirmed the Great Wall is not visible from space with the naked eye. At typical orbital altitude, it is too narrow to be distinguished.",
        "sources": ["apnews.com", "reuters.com"],
        "confidence": 0.95,
        "topic": "Science",
        "date": "2024-01-01"
    },
    {
        "claim": "Humans only use 10% of their brain",
        "verdict": "False",
        "summary": "Brain imaging studies show that over a day virtually all areas of the brain are active. This myth is not supported by neuroscience.",
        "sources": ["nih.gov", "nature.com"],
        "confidence": 0.98,
        "topic": "Science",
        "date": "2024-01-01"
    },
    {
        "claim": "Climate change is primarily caused by human activity",
        "verdict": "True",
        "summary": "Scientific consensus (97%+ of climate scientists) agrees that current climate change is primarily driven by human greenhouse gas emissions, supported by IPCC reports.",
        "sources": ["science.org", "nature.com", "who.int"],
        "confidence": 0.97,
        "topic": "Science",
        "date": "2024-01-01"
    },
    {
        "claim": "Lightning never strikes the same place twice",
        "verdict": "False",
        "summary": "Lightning frequently strikes the same place multiple times. Tall structures like the Empire State Building are struck dozens of times per year.",
        "sources": ["apnews.com", "reuters.com"],
        "confidence": 0.99,
        "topic": "Science",
        "date": "2024-01-01"
    },
    {
        "claim": "Eating carrots improves your night vision",
        "verdict": "Misleading",
        "summary": "Carrots contain beta-carotene which the body converts to Vitamin A, essential for eye health. However, excess carrots beyond a normal diet do not further improve vision in healthy individuals. This myth was partly WWII British propaganda.",
        "sources": ["nih.gov", "fullfact.org"],
        "confidence": 0.90,
        "topic": "Health",
        "date": "2024-01-01"
    },
]


class RAGKnowledgeBase:
    """Vector-based knowledge base for archived fact-checks and trusted sources."""

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="fact_checks",
            metadata={"hnsw:space": "cosine"}
        )
        self._seed_if_empty()

    def _seed_if_empty(self):
        """Seed the knowledge base with initial fact-checks if empty."""
        if self.collection.count() == 0:
            print("Seeding RAG knowledge base with initial fact-checks...")
            for i, fc in enumerate(SEED_FACT_CHECKS):
                text = f"Claim: {fc['claim']}\nVerdict: {fc['verdict']}\nSummary: {fc['summary']}"
                self.collection.add(
                    documents=[text],
                    ids=[f"seed_{i}"],
                    metadatas=[{
                        "claim": fc["claim"],
                        "verdict": fc["verdict"],
                        "confidence": fc["confidence"],
                        "topic": fc["topic"],
                        "sources": json.dumps(fc["sources"]),
                        "date": fc["date"],
                        "summary": fc["summary"]
                    }]
                )
            print(f"Seeded {len(SEED_FACT_CHECKS)} fact-checks.")

    def search_similar_claims(self, query: str, n_results: int = 5) -> list[dict]:
        """Search for similar previously verified claims."""
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, max(1, self.collection.count()))
        )

        hits = []
        if results and results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                similarity = 1 - dist  # cosine distance → similarity
                if similarity > 0.5:  # Only return reasonably similar
                    hits.append({
                        "document": doc,
                        "claim": meta.get("claim", ""),
                        "verdict": meta.get("verdict", ""),
                        "confidence": meta.get("confidence", 0),
                        "topic": meta.get("topic", ""),
                        "sources": json.loads(meta.get("sources", "[]")),
                        "summary": meta.get("summary", ""),
                        "date": meta.get("date", ""),
                        "similarity": round(similarity, 3)
                    })
        return hits

    def add_fact_check(self, claim: str, verdict: str, summary: str,
                       sources: list[str], confidence: float,
                       topic: str = "General") -> str:
        """Add a new verified fact-check to the archive."""
        doc_id = f"fc_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        text = f"Claim: {claim}\nVerdict: {verdict}\nSummary: {summary}"
        self.collection.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[{
                "claim": claim,
                "verdict": verdict,
                "confidence": confidence,
                "topic": topic,
                "sources": json.dumps(sources),
                "date": str(datetime.now().date()),
                "summary": summary
            }]
        )
        return doc_id

    def get_source_credibility(self, url: str) -> dict:
        """Return credibility info for a URL's domain."""
        for domain, info in TRUSTED_SOURCES.items():
            if domain in url:
                return {"domain": domain, **info}
        return {"domain": url, "credibility": 5, "type": "unknown",
                "notes": "Source not in trusted list — verify manually"}

    def get_all_trusted_sources(self) -> dict:
        """Return all trusted sources with credibility info."""
        return TRUSTED_SOURCES


# Singleton
_rag_instance: Optional[RAGKnowledgeBase] = None


def get_rag() -> RAGKnowledgeBase:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGKnowledgeBase()
    return _rag_instance
