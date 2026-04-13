"""
main.py — Punkt wejścia. Uruchom: python main.py

Krok po kroku:
  1. Wczytaj config.yaml
  2. Załaduj model i tokenizer z Hugging Face
  3. Stwórz agentów i sędziego
  4. Uruchom wybraną architekturę debaty
  5. Sędzia podsumowuje debatę
"""

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from agents import Agent, Judge
from architectures import ARCHITECTURES


def main():
    # 1. Wczytaj config
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Załaduj model (współdzielony przez wszystkich agentów)
    device = config.get("device", "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Ładowanie modelu: {config['model_name']} na {device}...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], dtype=dtype
    ).to(device)
    print("Model załadowany.\n")

    # 3. Stwórz agentów
    agents = [
        Agent(name=a["name"], system_prompt=a["system_prompt"], model=model, tokenizer=tokenizer)
        for a in config["agents"]
    ]
    judge = Judge(
        system_prompt=config["judge"]["system_prompt"], model=model, tokenizer=tokenizer
    )

    # 4. Uruchom debatę
    arch_name = config["architecture"]
    run_debate = ARCHITECTURES[arch_name]

    print(f"Architektura: {arch_name}")
    print(f"Temat: {config['topic']}")
    print(f"Rundy: {config['num_rounds']}")
    print(f"Agenci: {', '.join(a.name for a in agents)}")

    debate_log = run_debate(agents, judge, config["topic"], config["num_rounds"], config)

    # 5. Podsumowanie sędziego
    print(f"\n{'='*60}")
    print("  WERDYKT SĘDZIEGO")
    print(f"{'='*60}")
    verdict = judge.summarize(debate_log, config["topic"], config)
    print(f"\n{verdict}")


if __name__ == "__main__":
    main()
