
import os
import glob
import re
import pandas as pd
import string
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_decision(text):
    """Wyciąga decyzję (TAK/NIE) z wypowiedzi agenta."""
    if not isinstance(text, str): 
        return "BŁĄD_TYPU"
    match = re.search(r'DECYZJA:\s*[\[\*\s]*(TAK|NIE)', text, re.IGNORECASE)
    return match.group(1).upper() if match else "BŁĄD_FORMATU"


def calculate_flips(df, agent_col='Agent', round_col='Runda', decision_col='Decyzja'):
    """Wykrywa, kto zmienił zdanie (Number of Flips) i w której rundzie (Turn of Flips)."""
    flips_log = []
    
    for agent, group in df.groupby(agent_col):
        group = group.sort_values(by=round_col)
        poprzednia_decyzja = None
        
        for _, row in group.iterrows():
            obecna_decyzja = row[decision_col]
            

            if obecna_decyzja not in ["BŁĄD_FORMATU", "BŁĄD_TYPU"]:
                if poprzednia_decyzja is not None and obecna_decyzja != poprzednia_decyzja:
                    flips_log.append({
                        'Agent': agent,
                        'Runda_zmiany': row[round_col],
                        'Z_czego': poprzednia_decyzja,
                        'Na_co': obecna_decyzja
                    })
                poprzednia_decyzja = obecna_decyzja
                
    return flips_log


def calculate_convergence_round(df, round_col='Runda', decision_col='Decyzja', threshold=1.0):
    """Liczy pierwszą rundę, w której osiągnięto konsensus."""
    exclude_values = ['BŁĄD_FORMATU', 'BŁĄD_TYPU']
    df_sorted = df.sort_values(by=round_col)

    for runda, group in df_sorted.groupby(round_col):
        if len(group) == 0: 
            continue
        decision_shares = group[decision_col].value_counts(normalize=True)
        if not decision_shares.empty:
            if decision_shares.max() >= threshold and decision_shares.idxmax() not in exclude_values:
                return int(runda)
    return None


def calculate_lexical_diversity(text):
    """Oblicza TTR (Type-Token Ratio) - różnorodność słownictwa."""
    if not isinstance(text, str): return 0.0
    
    # usuwamy interpunkcję i zamieniamy na małe litery
    tekst_czysty = text.translate(str.maketrans('', '', string.punctuation)).lower()
    slowa = tekst_czysty.split()
    
    if len(slowa) == 0: return 0.0
    
    unikalne_slowa = set(slowa)
    return len(unikalne_slowa) / len(slowa)


# new
def cosine_similarity_text(a, b):
    """ Oblicza podobieństwo cosinusowe dwóch tekstów """
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0

    a = a.strip()
    b = b.strip()

    if len(a) < 2 or len(b) < 2:
        return 0.0

    vec = TfidfVectorizer().fit_transform([a, b])
    sim = cosine_similarity(vec[0], vec[1])[0][0]

    return sim


# new
def calculate_argument_metrics(df,
                               similarity_func,
                               agent_col='Agent',
                               round_col='Runda',
                               text_col='Odpowiedz'):
    """
    Coherence: similarity do poprzedniej wypowiedzi TEGO SAMEGO agenta
    Novelty: 1 - similarity do poprzedniej wypowiedzi INNEGO agenta
    """

    previous_by_agent = {}
    coherence_scores = []
    novelty_scores = []

    df_sorted = df.sort_values(by=[round_col])

    for _, row in df_sorted.iterrows():

        current_agent = row[agent_col]
        current_text = row[text_col]

        # COHERENCE
        if current_agent in previous_by_agent:

            own_previous = previous_by_agent[current_agent]

            coherence = similarity_func(
                own_previous,
                current_text
            )

        else: coherence = np.nan

        # NOVELTY
        other_texts = []

        for agent_name, text in previous_by_agent.items():

            if agent_name != current_agent:
                other_texts.append(text)

        if len(other_texts) > 0:

            similarities = [
                similarity_func(other_text, current_text)
                for other_text in other_texts
            ]

            avg_similarity = sum(similarities) / len(similarities)

            novelty = 1 - avg_similarity

        else: novelty = np.nan

        coherence_scores.append(coherence)
        novelty_scores.append(novelty)

        previous_by_agent[current_agent] = current_text

    return coherence_scores, novelty_scores


# new
def calculate_semantic_diversity(text, similarity_func=cosine_similarity_text):
    """Oblicza semantic diversity wypowiedzi: 1 - średnie podobieństwo cosinusowe między zdaniami"""

    if not isinstance(text, str):
        return 0.0

    # split na zdania
    sentences = re.split(r'[.!?]+', text)

    sentences = [
        s.strip()
        for s in sentences
        if isinstance(s, str) and len(s.strip()) > 1
    ]

    if len(sentences) < 2:
        return 0.0

    similarities = []

    # przejście po parach zdań
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = similarity_func(sentences[i], sentences[j])
            similarities.append(sim)

    avg_similarity = np.mean(similarities) if similarities else 0.0

    semantic_diversity = 1 - avg_similarity

    return semantic_diversity