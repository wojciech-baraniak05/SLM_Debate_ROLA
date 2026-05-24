"""
metrics_utils.py — Funkcje metryk kompatybilne z formatem JSON z pipeline.ipynb.

Format wejściowy (debate-N-result.json):
{
    "debate_log": [{"agent": str, "round": int, "text": str}, ...],
    "decision_log": [{"agent": str, "proposal": str, "<AgentName>": bool, ...}, ...],
    "topic": str,
    ...
}
"""

import re
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Wczytywanie danych z JSON (pipeline.ipynb → debate-N-result.json)
# ---------------------------------------------------------------------------

def load_debate_json(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Konwertuje surowy słownik z debate-N-result.json na dwa DataFrame:
      - df_debate: kolumny Agent, Runda, Odpowiedz
      - df_decision: kolumny Runda, Proponent, Agent, Proposal, Vote (bool)

    Zwraca (df_debate, df_decision).
    """
    # --- debate_log ---
    debate_rows = []
    for entry in data.get("debate_log", []):
        debate_rows.append({
            "Agent":    entry["agent"],
            "Runda":    entry["round"],
            "Odpowiedz": entry["text"],
        })
    df_debate = pd.DataFrame(debate_rows)

    # --- decision_log ---
    # Każdy wpis to słownik z kluczami: 'agent' (proponent propozycji),
    # 'proposal' (tekst propozycji) oraz nazwami agentów mapowanymi na bool.
    decision_rows = []
    for round_idx, entry in enumerate(data.get("decision_log", []), start=1):
        proposer  = entry.get("agent", "")
        proposal  = entry.get("proposal", "")
        # Głosy to wszystkie klucze poza 'agent' i 'proposal'
        for key, val in entry.items():
            if key in ("agent", "proposal"):
                continue
            decision_rows.append({
                "Runda_decyzji": round_idx,
                "Proponent":     proposer,
                "Agent":         key,
                "Proposal":      proposal,
                "Vote":          bool(val),
            })
    df_decision = pd.DataFrame(decision_rows)

    return df_debate, df_decision


# ---------------------------------------------------------------------------
# Konsensus i flips (operują na df_decision)
# ---------------------------------------------------------------------------

def calculate_consensus_reached(df_decision: pd.DataFrame,
                                 threshold: float = 0.66) -> int | None:
    """
    Zwraca numer rundy decyzyjnej, w której osiągnięto konsensus
    (odsetek głosów YES >= threshold), lub None.
    """
    if df_decision.empty:
        return None

    for runda, group in df_decision.groupby("Runda_decyzji"):
        if group.empty:
            continue
        n         = len(group)
        yes_ratio = group["Vote"].mean()
        no_ratio  = 1.0 - yes_ratio
        if yes_ratio >= threshold or no_ratio >= threshold:
            return int(runda)
    return None


def calculate_flips(df_decision: pd.DataFrame) -> list[dict]:
    """
    Wykrywa zmiany głosu (TRUE→FALSE lub FALSE→TRUE) dla każdego agenta
    w kolejnych rundach decyzyjnych.

    Zwraca listę słowników: Agent, Runda_zmiany, Z_czego (bool), Na_co (bool).
    """
    flips_log = []

    for agent, group in df_decision.groupby("Agent"):
        group = group.sort_values("Runda_decyzji")
        prev_vote = None

        for _, row in group.iterrows():
            curr_vote = row["Vote"]
            if prev_vote is not None and curr_vote != prev_vote:
                flips_log.append({
                    "Agent":         agent,
                    "Runda_zmiany":  row["Runda_decyzji"],
                    "Z_czego":       prev_vote,
                    "Na_co":         curr_vote,
                })
            prev_vote = curr_vote

    return flips_log


# ---------------------------------------------------------------------------
# Metryki językowe (operują na df_debate["Odpowiedz"])
# ---------------------------------------------------------------------------

def calculate_lexical_diversity(text: str) -> float:
    """TTR (Type-Token Ratio) — różnorodność słownictwa."""
    if not isinstance(text, str):
        return 0.0
    clean = text.translate(str.maketrans("", "", string.punctuation)).lower()
    words = clean.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def cosine_similarity_text(a: str, b: str) -> float:
    """Podobieństwo cosinusowe dwóch tekstów (TF-IDF)."""
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    a, b = a.strip(), b.strip()
    if len(a) < 2 or len(b) < 2:
        return 0.0
    vec = TfidfVectorizer().fit_transform([a, b])
    return float(cosine_similarity(vec[0], vec[1])[0][0])


def calculate_argument_metrics(
    df: pd.DataFrame,
    similarity_func=cosine_similarity_text,
    agent_col: str = "Agent",
    round_col: str = "Runda",
    text_col: str = "Odpowiedz",
) -> tuple[list, list]:
    """
    Coherence: podobieństwo do poprzedniej wypowiedzi TEGO SAMEGO agenta.
    Novelty:   1 - średnie podobieństwo do poprzednich wypowiedzi INNYCH agentów.

    Zwraca (coherence_scores, novelty_scores) — listy o tej samej długości co df.
    """
    previous_by_agent: dict[str, str] = {}
    coherence_scores, novelty_scores = [], []

    for _, row in df.sort_values(by=round_col).iterrows():
        agent = row[agent_col]
        text  = row[text_col]

        # Coherence
        if agent in previous_by_agent:
            coherence = similarity_func(previous_by_agent[agent], text)
        else:
            coherence = np.nan

        # Novelty
        other_texts = [t for a, t in previous_by_agent.items() if a != agent]
        if other_texts:
            sims = [similarity_func(t, text) for t in other_texts]
            novelty = 1.0 - float(np.mean(sims))
        else:
            novelty = np.nan

        coherence_scores.append(coherence)
        novelty_scores.append(novelty)
        previous_by_agent[agent] = text

    return coherence_scores, novelty_scores


def calculate_semantic_diversity(
    text: str,
    similarity_func=cosine_similarity_text,
) -> float:
    """
    Semantic diversity = 1 - średnie podobieństwo cosinusowe między zdaniami.
    """
    if not isinstance(text, str):
        return 0.0
    sentences = [
        s.strip()
        for s in re.split(r"[.!?]+", text)
        if isinstance(s, str) and len(s.strip()) > 1
    ]
    if len(sentences) < 2:
        return 0.0
    sims = [
        similarity_func(sentences[i], sentences[j])
        for i in range(len(sentences))
        for j in range(i + 1, len(sentences))
    ]
    return float(1.0 - np.mean(sims)) if sims else 0.0


# ---------------------------------------------------------------------------
# Order importance (wpływ kolejności startu na wynik konsensusu)
# ---------------------------------------------------------------------------

def calculate_order_importance(records: list[dict]) -> float:
    """
    Przyjmuje listę słowników:
        {"first_agent_vote": bool | None, "final_consensus_vote": bool | None}
    Zwraca odsetek debat, w których głos pierwszego agenta z rundy 1
    był zgodny z końcowym konsensusem.
    """
    valid = [
        r for r in records
        if r.get("first_agent_vote") is not None
        and r.get("final_consensus_vote") is not None
    ]
    if not valid:
        return 0.0
    matches = sum(
        1 for r in valid
        if r["first_agent_vote"] == r["final_consensus_vote"]
    )
    return (matches / len(valid)) * 100.0
