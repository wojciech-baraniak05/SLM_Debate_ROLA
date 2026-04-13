"""
architectures.py — Trzy architektury wymiany informacji w debacie.

Każda funkcja przyjmuje tych samych argumentów i zwraca debate_log:
    debate_log = [{"agent": str, "round": int, "text": str}, ...]

Architektury:
    round_robin  — wszyscy widzą całą historię, stała kolejność
    relay        — każdy widzi TYLKO wypowiedź poprzedniego (głuchy telefon)
    free_for_all — wszyscy widzą całą historię, losowa kolejność w każdej rundzie
"""

import random


def round_robin(agents, judge, topic, num_rounds, config):
    """Każdy agent odpowiada po kolei. Wszyscy widzą całą historię."""
    debate_log = []
    history = [f"Temat debaty: {topic}"]

    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"  RUNDA {round_num}")
        print(f"{'='*60}")

        for agent in agents:
            response = agent.respond(history, config)
            debate_log.append({"agent": agent.name, "round": round_num, "text": response})
            history.append(f"{agent.name}: {response}")
            print(f"\n[{agent.name}]: {response}")

    return debate_log


def relay(agents, judge, topic, num_rounds, config):
    """Łańcuch — każdy agent widzi TYLKO wypowiedź poprzedniego agenta + temat."""
    debate_log = []
    previous_response = f"Temat debaty: {topic}"

    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"  RUNDA {round_num}")
        print(f"{'='*60}")

        for agent in agents:
            # Agent widzi tylko temat + ostatnią wypowiedź
            history = [f"Temat debaty: {topic}", previous_response]
            response = agent.respond(history, config)
            debate_log.append({"agent": agent.name, "round": round_num, "text": response})
            previous_response = f"{agent.name}: {response}"
            print(f"\n[{agent.name}]: {response}")

    return debate_log


def free_for_all(agents, judge, topic, num_rounds, config):
    """Jak round_robin, ale kolejność agentów jest losowa w każdej rundzie."""
    debate_log = []
    history = [f"Temat debaty: {topic}"]

    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"  RUNDA {round_num}")
        print(f"{'='*60}")

        shuffled = list(agents)
        random.shuffle(shuffled)
        order = ", ".join(a.name for a in shuffled)
        print(f"  Kolejność: {order}")

        for agent in shuffled:
            response = agent.respond(history, config)
            debate_log.append({"agent": agent.name, "round": round_num, "text": response})
            history.append(f"{agent.name}: {response}")
            print(f"\n[{agent.name}]: {response}")

    return debate_log


# Mapowanie nazw z configu na funkcje
ARCHITECTURES = {
    "round_robin": round_robin,
    "relay": relay,
    "free_for_all": free_for_all,
}
