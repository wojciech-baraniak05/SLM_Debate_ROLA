"""
analiza.py — Skrypt do ekstrakcji metryk z przeprowadzonych debat.
Uruchom: python analiza.py
"""

import os
import glob
import re
import pandas as pd
import string
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from metrics_utils import (
    extract_decision, calculate_flips, calculate_convergence_round,
    calculate_lexical_diversity, cosine_similarity_text,
    calculate_argument_metrics, calculate_semantic_diversity
)


def main():
    folder_wynikow = "wyniki"
    pliki_csv = glob.glob(f"{folder_wynikow}/*.csv")
    
    if not pliki_csv:
        print(f"Brak plików .csv w folderze '{folder_wynikow}'. Uruchom najpierw main.py.")
        return

    print(f"Znaleziono {len(pliki_csv)} debat do przeanalizowania.\n")
    
    wyniki_konsensusu = []
    wszystkie_flips = []
    wszystkie_dane = []
    # new
    dane_pozycji_agentow = []

    for sciezka in pliki_csv:
        df = pd.read_csv(sciezka)

        # coś mi nie działało z nazwą więc tu ujednolicam
        for col in ['Latency', 'latency']:
            if col in df.columns:
                df.rename(columns={col: 'Opoznienie'}, inplace=True)
        
        # Obliczanie metryk na podstawie debaty
        if 'Odpowiedz' in df.columns:
            df['Decyzja'] = df['Odpowiedz'].apply(extract_decision)
            df['Liczba_Slow'] = df['Odpowiedz'].apply(lambda x: len(str(x).split()))
            df['Roznorodnosc_TTR'] = df['Odpowiedz'].apply(calculate_lexical_diversity)
            # new
            coherence_scores, novelty_scores = calculate_argument_metrics(df, cosine_similarity_text)
            df['Spojnosc'] = coherence_scores
            df['Nowe_argumenty'] = novelty_scores
            df['Roznorodnosc_Semantyczna'] = df['Odpowiedz'].apply(calculate_semantic_diversity)
        else:
            continue
            
        runda_kons = calculate_convergence_round(df)
        wyniki_konsensusu.append(runda_kons)
        flips = calculate_flips(df)
        wszystkie_flips.extend(flips)
        
        wszystkie_dane.append(df)
        
        nazwa = os.path.basename(sciezka)
        print(f"{nazwa}: Konsensus: {runda_kons if runda_kons else 'Brak'} | Zmian zdania: {len(flips)}")

        # new - analiza wpływu kolejności
        df_runda_1 = df[df['Runda'] == 1]
        if not df_runda_1.empty:
            lider_startowy = df_runda_1.iloc[0]['Agent']
            decyzja_poczatkowa = df_runda_1.iloc[0]['Decyzja']
        else:
            lider_startowy = None
            decyzja_poczatkowa = None

        koncowy_konsensus = None
        if runda_kons:
            df_konsensus = df[df['Runda'] == runda_kons]
            if not df_konsensus.empty:
                # najczęstszą decyzję z rundy końcowej (powinna być jedna wspólna)
                koncowy_konsensus = df_konsensus['Decyzja'].value_counts().idxmax()

        dane_pozycji_agentow.append({
            'Plik': nazwa,
            'Lider_Startowy': lider_startowy,
            'Decyzja_Poczatkowa': decyzja_poczatkowa,
            'Koncowy_Konsensus': koncowy_konsensus
        })

    udane_debaty = [r for r in wyniki_konsensusu if r is not None]

    # --- Obliczanie order importance ---
    sukcesy_lidera_startowego = 0
    debaty_z_konsensusem = 0

    for debata in dane_pozycji_agentow:
        if debata['Koncowy_Konsensus'] and debata['Koncowy_Konsensus'] not in ['BŁĄD_FORMATU', 'BŁĄD_TYPU']:
            if debata['Decyzja_Poczatkowa'] and debata['Decyzja_Poczatkowa'] not in ['BŁĄD_FORMATU', 'BŁĄD_TYPU']:
                debaty_z_konsensusem += 1

                # Jeśli pierwsza decyzja lidera stała się końcowym konsensusem debaty:
                if debata['Decyzja_Poczatkowa'] == debata['Koncowy_Konsensus']:
                    sukcesy_lidera_startowego += 1

    if debaty_z_konsensusem > 0:
        order_importance_rate = (sukcesy_lidera_startowego / debaty_z_konsensusem) * 100
    else:
        order_importance_rate = 0.0




    print(f"\n{'='*40}\n RAPORT KOŃCOWY:\n{'='*40}")

    # Przeniesione wyżej
    if wszystkie_dane:
        df_global = pd.concat(wszystkie_dane, ignore_index=True)
    else:
        print("Brak poprawnych danych do analizy.")
        return

    print(f"Całkowita liczba eksperymentów: {len(pliki_csv)}")
    print(f"Liczba debat z osiągniętym konsensusem: {len(udane_debaty)}")
    if udane_debaty:
        srednia_runda = sum(udane_debaty) / len(udane_debaty)
        success_rate = (len(udane_debaty) / len(pliki_csv)) * 100
        print(f"Wskaźnik sukcesu (Success Rate): {success_rate:.1f}%")
        print(f"Średnia runda konwergencji: {srednia_runda:.2f}")


    print("\n--- STATYSTYKI ZMIANY ZDANIA (FLIPS) ---")

    if wszystkie_flips:
        df_flips = pd.DataFrame(wszystkie_flips)
        print("Kto najczęściej zmieniał zdanie:")
        print(df_flips['Agent'].value_counts().to_string())
        print(f"\nŚrednia runda, w której ktoś uległ: {df_flips['Runda_zmiany'].mean():.1f}")
    else:
        print("Żaden agent nie zmienił zdania w żadnej debacie.")


    print("\n--- STATYSTYKI WYPOWIEDZI (TOKENY I SŁOWNICTWO) ---")

    print("Średnia długość wypowiedzi:")
    srednia_dlugosc = df_global.groupby('Agent')['Liczba_Slow'].mean()
    for agent, wartosc in srednia_dlugosc.items():
        print(f" - {agent}: {wartosc:.1f} słów")
            
    print("\nŚrednia różnorodność leksykalna (TTR):")
    srednia_ttr = df_global.groupby('Agent')['Roznorodnosc_TTR'].mean()
    for agent, wartosc in srednia_ttr.items():
        print(f" - {agent}: {wartosc:.3f}")


    # new
    if 'Opoznienie' in df_global.columns:

        print("\n--- OPÓŹNIENIE (CZAS GENEROWANIA ODPOWIEDZI) ---")

        print('Średni czas wypowiedzi:')

        sredni_latency = df_global.groupby('Agent')['Opoznienie'].mean()

        for agent, wartosc in sredni_latency.items():
            print(f" - {agent}: {wartosc:.2f} s")

        liczba_eksperymentow = len(pliki_csv)
        sredni_czas_eksperymentu = df_global['Opoznienie'].sum() / liczba_eksperymentow

        print(f"\n - Jeden eksperyment: {sredni_czas_eksperymentu:.2f} s")


    # new
    print("\n--- SPÓJNOŚĆ I NOWOŚĆ ARGUMENTACJI ---")

    print("\nŚrednia coherence:")

    srednia_coherence = (
        df_global
        .groupby('Agent')['Spojnosc']
        .mean()
    )

    for agent, wartosc in srednia_coherence.items():
        print(f" - {agent}: {wartosc:.3f}")

    print("\nŚrednia novelty:")

    srednia_novelty = (
        df_global
        .groupby('Agent')['Nowe_argumenty']
        .mean()
    )

    for agent, wartosc in srednia_novelty.items():
        print(f" - {agent}: {wartosc:.3f}")


    # new
    print("\n--- METAMETRYKI EXPERYMENTU ---")
    print(f"Wpływ kolejności startowej (Order Importance): {order_importance_rate:.1f}%")    

    # WARIANCJA METRYK
    # średnie z debat dla każdego agenta
    metryki_list = []
    for df_debata in wszystkie_dane:
        for agent in df_global['Agent'].unique():
            df_agent = df_debata[df_debata['Agent'] == agent]
            if not df_agent.empty:
                metryki_list.append({
                    'Agent': agent,
                    'Liczba_Slow': df_agent['Liczba_Slow'].mean(),
                    'Spojnosc': df_agent['Spojnosc'].mean(),
                    'Nowe_argumenty': df_agent['Nowe_argumenty'].mean(),
                    'Opoznienie': df_agent['Opoznienie'].mean() if 'Opoznienie' in df_debata.columns else np.nan
                })
                    
    if metryki_list:
        df_res = pd.DataFrame(metryki_list)
            
        print("\nWariancja średniej długości wypowiedzi:")
        var_slow = df_res.groupby('Agent')['Liczba_Slow'].var(ddof=1)
        for agent, wartosc in var_slow.items():
            print(f" - {agent}: {wartosc:.2f}")
                
        print("\nWariancja średniej spójności (Coherence):")
        var_coh = df_res.groupby('Agent')['Spojnosc'].var(ddof=1)
        for agent, wartosc in var_coh.items():
            print(f" - {agent}: {wartosc:.5f}")
                
        print("\nWariancja średniej nowości (Novelty):")
        var_nov = df_res.groupby('Agent')['Nowe_argumenty'].var(ddof=1)
        for agent, wartosc in var_nov.items():
            print(f" - {agent}: {wartosc:.5f}")

        if 'Opoznienie' in df_global.columns:
            print("\nWariancja średniego opóźnienia:")
            var_lat = df_res.groupby('Agent')['Opoznienie'].var(ddof=1)
            for agent, wartosc in var_lat.items():
                print(f" - {agent}: {wartosc:.2f}")


if __name__ == "__main__":
    main()
