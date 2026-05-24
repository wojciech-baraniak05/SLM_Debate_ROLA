"""
analiza.py — Analiza metryk z debat zapisanych przez pipeline.ipynb.

Oczekiwany format pliku: data/debate-N-result.json
Uruchomienie: python analiza.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from metrics_utils import (
    load_debate_json,
    calculate_consensus_reached,
    calculate_flips,
    calculate_lexical_diversity,
    cosine_similarity_text,
    calculate_argument_metrics,
    calculate_semantic_diversity,
    calculate_order_importance,
)


DEBATE_DIR   = "data"
CONSENSUS_THRESHOLD = 0.66


def _enrich_debate_df(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje kolumny metryk do df_debate."""
    df = df.copy()
    df["Liczba_Slow"]         = df["Odpowiedz"].apply(lambda x: len(str(x).split()))
    df["Roznorodnosc_TTR"]    = df["Odpowiedz"].apply(calculate_lexical_diversity)
    df["Roznorodnosc_Sem"]    = df["Odpowiedz"].apply(calculate_semantic_diversity)

    coh, nov = calculate_argument_metrics(df, cosine_similarity_text)
    df["Spojnosc"]            = coh
    df["Nowe_argumenty"]      = nov
    return df


def main():
    pattern = os.path.join(DEBATE_DIR, "debate-*-result.json")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"Brak plików JSON w folderze '{DEBATE_DIR}'.")
        print("Upewnij się, że pipeline.ipynb zapisał wyniki za pomocą save_debate_result().")
        return

    print(f"Znaleziono {len(files)} pliki debat.\n")

    # Agregaty
    all_df_debate:   list[pd.DataFrame] = []
    all_df_decision: list[pd.DataFrame] = []
    consensus_rounds: list[int | None]  = []
    all_flips:        list[dict]        = []
    order_records:    list[dict]        = []

    for path in files:
        fname = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        df_debate, df_decision = load_debate_json(data)

        if df_debate.empty:
            print(f"  [{fname}] — brak danych w debate_log, pomijam.")
            continue

        df_debate = _enrich_debate_df(df_debate)

        # Konsensus
        runda_kons = (
            calculate_consensus_reached(df_decision, CONSENSUS_THRESHOLD)
            if not df_decision.empty else None
        )
        consensus_rounds.append(runda_kons)

        # Flips (zmiany głosu w rundach decyzyjnych)
        flips = calculate_flips(df_decision) if not df_decision.empty else []
        all_flips.extend(flips)

        # Order importance — pierwsza wypowiedź rundy 1 + końcowy konsensus
        first_vote      = None
        final_consensus = None

        if not df_decision.empty:
            r1 = df_decision[df_decision["Runda_decyzji"] == 1]
            if not r1.empty:
                first_vote = bool(r1.sort_values("Agent").iloc[0]["Vote"])

            if runda_kons is not None:
                rk = df_decision[df_decision["Runda_decyzji"] == runda_kons]
                if not rk.empty:
                    # konsensus = głosowała większość tak samo
                    vc = rk["Vote"].value_counts()
                    final_consensus = bool(vc.idxmax())

        order_records.append({
            "first_agent_vote":    first_vote,
            "final_consensus_vote": final_consensus,
        })

        all_df_debate.append(df_debate)
        all_df_decision.append(df_decision)

        kons_str = str(runda_kons) if runda_kons else "brak"
        print(f"[{fname}] konsensus: runda {kons_str} | flipy: {len(flips)}")

    if not all_df_debate:
        print("\nBrak poprawnych danych do analizy.")
        return

    df_global   = pd.concat(all_df_debate,   ignore_index=True)
    udane       = [r for r in consensus_rounds if r is not None]
    n_ekspery   = len(files)

    # -----------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(" RAPORT KOŃCOWY")
    print(f"{'='*50}")

    print(f"\nLiczba eksperymentów:             {n_ekspery}")
    print(f"Debaty z konsensusem:             {len(udane)}")

    if udane:
        sr = (len(udane) / n_ekspery) * 100
        print(f"Success Rate:                     {sr:.1f}%")
        print(f"Śr. runda konwergencji:           {np.mean(udane):.2f}")

    # --- Flips ---
    print(f"\n--- ZMIANY GŁOSU (FLIPS) ---")
    if all_flips:
        df_flips = pd.DataFrame(all_flips)
        print("Kto najczęściej zmieniał zdanie:")
        print(df_flips["Agent"].value_counts().to_string())
        print(f"\nŚr. runda zmiany:  {df_flips['Runda_zmiany'].mean():.2f}")
    else:
        print("Żaden agent nie zmienił zdania.")

    # --- Długość i słownictwo ---
    print(f"\n--- WYPOWIEDZI: DŁUGOŚĆ I SŁOWNICTWO ---")
    print("Średnia liczba słów:")
    for agent, val in df_global.groupby("Agent")["Liczba_Slow"].mean().items():
        print(f"  {agent}: {val:.1f}")

    print("\nŚredni TTR (różnorodność leksykalna):")
    for agent, val in df_global.groupby("Agent")["Roznorodnosc_TTR"].mean().items():
        print(f"  {agent}: {val:.3f}")

    print("\nŚrednia różnorodność semantyczna:")
    for agent, val in df_global.groupby("Agent")["Roznorodnosc_Sem"].mean().items():
        print(f"  {agent}: {val:.3f}")

    # --- Spójność i nowość ---
    print(f"\n--- ARGUMENTACJA: SPÓJNOŚĆ I NOWOŚĆ ---")
    print("Coherence (spójność z poprzednią własną wypowiedzią):")
    for agent, val in df_global.groupby("Agent")["Spojnosc"].mean().items():
        print(f"  {agent}: {val:.3f}")

    print("\nNovelty (nowość względem innych agentów):")
    for agent, val in df_global.groupby("Agent")["Nowe_argumenty"].mean().items():
        print(f"  {agent}: {val:.3f}")

    # --- Order importance ---
    order_rate = calculate_order_importance(order_records)
    print(f"\n--- METAMETRYKI ---")
    print(f"Wpływ kolejności startowej (Order Importance): {order_rate:.1f}%")

    # --- Wariancja między debatami ---
    metryki_list = []
    for df_d in all_df_debate:
        for agent in df_global["Agent"].unique():
            sub = df_d[df_d["Agent"] == agent]
            if sub.empty:
                continue
            metryki_list.append({
                "Agent":          agent,
                "Liczba_Slow":    sub["Liczba_Slow"].mean(),
                "Spojnosc":       sub["Spojnosc"].mean(),
                "Nowe_argumenty": sub["Nowe_argumenty"].mean(),
            })

    if metryki_list:
        df_res = pd.DataFrame(metryki_list)
        print(f"\n--- WARIANCJA MIĘDZY DEBATAMI ---")

        print("Wariancja liczby słów:")
        for agent, val in df_res.groupby("Agent")["Liczba_Slow"].var(ddof=1).items():
            print(f"  {agent}: {val:.2f}")

        print("\nWariancja spójności:")
        for agent, val in df_res.groupby("Agent")["Spojnosc"].var(ddof=1).items():
            print(f"  {agent}: {val:.5f}")

        print("\nWariancja nowości:")
        for agent, val in df_res.groupby("Agent")["Nowe_argumenty"].var(ddof=1).items():
            print(f"  {agent}: {val:.5f}")

    # --- Statystyki głosowania ---
    if all_df_decision:
        df_dec_global = pd.concat(all_df_decision, ignore_index=True)
        if not df_dec_global.empty:
            print(f"\n--- GŁOSOWANIE W FAZIE KONSENSUSU ---")
            vote_rates = df_dec_global.groupby("Agent")["Vote"].mean()
            print("Odsetek głosów YES per agent:")
            for agent, val in vote_rates.items():
                print(f"  {agent}: {val:.1%}")


if __name__ == "__main__":
    main()
