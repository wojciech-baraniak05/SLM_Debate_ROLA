# Multi-Agent Debate z małymi modelami (SLM)

Minimalistyczne repozytorium do eksperymentów z debatą wieloagentową.
Agenci korzystają z małych modeli językowych (TinyLlama, Bielik) uruchamianych lokalnie.

## Szybki start

```bash
pip install -r requirements.txt
python main.py
```

Przy pierwszym uruchomieniu model zostanie pobrany z Hugging Face (~2 GB dla TinyLlama).

## Struktura projektu

```
config.yaml        ← Wszystkie parametry do edycji (prompty, model, architektura)
main.py            ← Punkt wejścia — ładuje config, model, uruchamia debatę
agents.py          ← Klasa Agent (generuje odpowiedź) i Judge (podsumowuje)
architectures.py   ← Trzy architektury wymiany informacji
```

## Co zmieniać w `config.yaml`

### Model
```yaml
model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# lub:
model_name: "speakleash/Bielik-1B-Instruct-v0.1"
```

### Parametry generowania
```yaml
temperature: 0.7     # niższa = bardziej deterministyczny
do_sample: true       # false = greedy (zawsze ten sam wynik)
max_new_tokens: 256   # maks. długość odpowiedzi
```

### Architektura debaty
```yaml
architecture: "round_robin"   # wszyscy widzą całą historię
architecture: "relay"         # każdy widzi TYLKO poprzedniego (głuchy telefon)
architecture: "free_for_all"  # jak round_robin, ale losowa kolejność
```

### Agenci i ich prompty
Dodaj, usuń lub zmień agentów w sekcji `agents`. Każdy ma `name` i `system_prompt`.

## Jak działa debata — krok po kroku

1. `main.py` wczytuje `config.yaml` i ładuje model z Hugging Face
2. Tworzy obiekty `Agent` (uczestnicy) i `Judge` (sędzia) — wszyscy współdzielą jeden model
3. Wybrana architektura z `architectures.py` steruje kto widzi jakie informacje:
   - **round_robin**: agent dostaje pełną historię wszystkich wypowiedzi
   - **relay**: agent dostaje tylko temat + ostatnią wypowiedź
   - **free_for_all**: jak round_robin, ale kolejność losowa
4. W każdej rundzie agenci generują odpowiedzi przez `Agent.respond()`
5. Na końcu `Judge.summarize()` ocenia całą debatę

## Eksperymenty do przeprowadzenia

1. **Temperatura**: porównaj wyniki z `temperature: 0.0` vs `1.0`
2. **Architektura**: jak zmienia się jakość argumentów w relay vs round_robin?
3. **Liczba rund**: czy więcej rund = lepsza debata?
4. **Prompty systemowe**: jak zmiana roli agenta wpływa na argumenty?
5. **Model**: TinyLlama vs Bielik — różnice w stylu i jakości?

## Wymagania

- Python 3.10+
- GPU z ~4 GB VRAM (lub CPU — wolniej, ale działa)
