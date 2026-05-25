"""
analiza.py — Analiza metryk z debat zapisanych przez pipeliny.

Struktura folderów (tworzona przez pipeline):
    data/
        config_1_proponent_opponent_opponent/
            debate-1-result.json
            debate-2-result.json
            ...
        config_2_opponent_opponent_proponent/
            ...
        ...

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


BASE_DIR            = "data"
CONSENSUS_THRESHOLD = 0.66


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enrich_debate_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Liczba_Slow"]      = df["Odpowiedz"].apply(lambda x: len(str(x).split()))
    df["Roznorodnosc_TTR"] = df["Odpowiedz"].apply(calculate_lexical_diversity)
    df["Roznorodnosc_Sem"] = df["Odpowiedz"].apply(calculate_semantic_diversity)
    coh, nov = calculate_argument_metrics(df, cosine_similarity_text)
    df["Spojnosc"]         = coh
    df["Nowe_argumenty"]   = nov
    return df


def _process_config_folder(folder_path: str) -> dict:
    """
    Wczytuje wszystkie debate-N-result.json z jednego podfolderu konfiguracji.
    Zwraca słownik z danymi gotowymi do raportu.
    """
    files = sorted(glob.glob(os.path.join(folder_path, "debate-*-result.json")))
    config_name = os.path.basename(folder_path)

    all_df_debate    = []
    all_df_decision  = []
    consensus_rounds = []
    all_flips        = []
    order_records    = []

    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        df_debate, df_decision = load_debate_json(data)

        if df_debate.empty:
            continue

        df_debate = _enrich_debate_df(df_debate)

        runda_kons = (
            calculate_consensus_reached(df_decision, CONSENSUS_THRESHOLD)
            if not df_decision.empty else None
        )
        consensus_rounds.append(runda_kons)

        flips = calculate_flips(df_decision) if not df_decision.empty else []
        all_flips.extend(flips)

        # Order importance — używamy Speaking_Order z load_debate_json,
        # żeby wziąć głos agenta który mówił PIERWSZY w debacie,
        # a nie pierwszego alfabetycznie (działa dla 2 i 3 agentów).
        first_vote      = None
        final_consensus = None
        if not df_decision.empty:
            r1 = df_decision[df_decision["Runda_decyzji"] == 1]
            if not r1.empty:
                first_speaker = r1.sort_values("Speaking_Order").iloc[0]
                first_vote = bool(first_speaker["Vote"])
            if runda_kons is not None:
                    final_consensus = True  # konsensus = wystarczająco dużo YES

        order_records.append({
            "first_agent_vote":     first_vote,
            "final_consensus_vote": final_consensus,
        })

        all_df_debate.append(df_debate)
        all_df_decision.append(df_decision)

    return {
        "config_name":     config_name,
        "n_debates":       len(files),
        "df_debate_list":  all_df_debate,
        "df_dec_list":     all_df_decision,
        "consensus_rounds": consensus_rounds,
        "flips":           all_flips,
        "order_records":   order_records,
    }


def _print_config_report(result: dict):
    config_name      = result["config_name"]
    n_debates        = result["n_debates"]
    consensus_rounds = result["consensus_rounds"]
    all_flips        = result["flips"]
    order_records    = result["order_records"]
    all_df_debate    = result["df_debate_list"]
    all_df_decision  = result["df_dec_list"]

    udane = [r for r in consensus_rounds if r is not None]

    print(f"\n{'='*60}")
    print(f"  {config_name}")
    print(f"{'='*60}")
    print(f"Debaty:                    {n_debates}")
    print(f"Konsensus:                 {len(udane)}/{n_debates}", end="")
    if n_debates:
        print(f"  ({len(udane)/n_debates*100:.0f}%)", end="")
    print()

    if udane:
        print(f"Śr. runda konwergencji:    {np.mean(udane):.2f}")

    if not all_df_debate:
        print("  (brak danych)")
        return

    df_global = pd.concat(all_df_debate, ignore_index=True)

    # Flips
    if all_flips:
        df_flips = pd.DataFrame(all_flips)
        print(f"\nZmiany zdania (flips):")
        for agent, cnt in df_flips["Agent"].value_counts().items():
            print(f"  {agent}: {cnt}x")
    else:
        print("\nZmiany zdania: brak")

    # Długość wypowiedzi
    print("\nŚr. liczba słów:")
    for agent, val in df_global.groupby("Agent")["Liczba_Slow"].mean().items():
        print(f"  {agent}: {val:.1f}")

    # TTR
    print("\nŚr. TTR (różnorodność leksykalna):")
    for agent, val in df_global.groupby("Agent")["Roznorodnosc_TTR"].mean().items():
        print(f"  {agent}: {val:.3f}")

    # Semantic diversity
    print("\nŚr. różnorodność semantyczna:")
    for agent, val in df_global.groupby("Agent")["Roznorodnosc_Sem"].mean().items():
        print(f"  {agent}: {val:.3f}")

    # Coherence / Novelty
    print("\nŚr. coherence (spójność):")
    for agent, val in df_global.groupby("Agent")["Spojnosc"].mean().items():
        print(f"  {agent}: {val:.3f}")

    print("\nŚr. novelty (nowość):")
    for agent, val in df_global.groupby("Agent")["Nowe_argumenty"].mean().items():
        print(f"  {agent}: {val:.3f}")

    # Głosowanie
    if all_df_decision:
        df_dec = pd.concat(all_df_decision, ignore_index=True)
        if not df_dec.empty:
            print("\nOdsetek głosów YES:")
            for agent, val in df_dec.groupby("Agent")["Vote"].mean().items():
                print(f"  {agent}: {val:.1%}")

    # Order importance
    order_rate = calculate_order_importance(order_records)
    print(f"\nOrder importance:          {order_rate:.1f}%")


def _print_summary_report(all_results: list[dict]):
    print(f"\n\n{'#'*60}")
    print(f"  RAPORT ZBIORCZY — wszystkie konfiguracje")
    print(f"{'#'*60}")

    rows = []
    for result in all_results:
        udane = [r for r in result["consensus_rounds"] if r is not None]
        n     = result["n_debates"]
        rows.append({
            "Konfiguracja":     result["config_name"],
            "Debaty":           n,
            "Konsensus":        len(udane),
            "Success Rate":     f"{len(udane)/n*100:.0f}%" if n else "—",
            "Śr. runda kons.":  f"{np.mean(udane):.2f}" if udane else "—",
            "Flips":            len(result["flips"]),
            "Order Importance": f"{calculate_order_importance(result['order_records']):.1f}%",
        })

    df_summary = pd.DataFrame(rows).set_index("Konfiguracja")
    print(df_summary.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.isdir(BASE_DIR):
        print(f"Folder '{BASE_DIR}' nie istnieje.")
        print("Upewnij się, że pipeline.py zapisał wyniki.")
        return

    subfolders = sorted(
        [d for d in os.scandir(BASE_DIR) if d.is_dir()],
        key=lambda d: d.name,
    )

    if not subfolders:
        print(f"Brak podfolderów w '{BASE_DIR}'.")
        return

    print(f"Znaleziono {len(subfolders)} konfiguracji: "
          f"{', '.join(d.name for d in subfolders)}")

    all_results = []
    for d in subfolders:
        result = _process_config_folder(d.path)
        _print_config_report(result)
        all_results.append(result)

    if len(all_results) > 1:
        _print_summary_report(all_results)


if __name__ == "__main__":
    main()
