"""
agents.py — Agent i Judge do multi-agent debate.

Każdy Agent ma swój system_prompt, ale współdzieli model z innymi agentami.
"""

import torch


def _generate(model, tokenizer, messages, config):
    """Wspólna funkcja generowania odpowiedzi z chat template.

    Args:
        model: model HF
        tokenizer: tokenizer HF
        messages: lista dict {"role": ..., "content": ...}
        config: dict z parametrami generowania
    Returns:
        str — wygenerowany tekst
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.get("max_new_tokens", 256),
            max_length=None,  # wyłącz domyślny limit modelu
            temperature=config.get("temperature", 0.7),
            do_sample=config.get("do_sample", True),
            pad_token_id=tokenizer.eos_token_id,
        )

    # Dekodujemy tylko nowo wygenerowane tokeny
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class Agent:
    """Jeden uczestnik debaty."""

    def __init__(self, name, system_prompt, model, tokenizer):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.tokenizer = tokenizer

    def respond(self, conversation_history, config):
        """Generuje odpowiedź na podstawie historii rozmowy.

        Args:
            conversation_history: lista stringów — dotychczasowe wypowiedzi
            config: dict z parametrami generowania
        Returns:
            str — odpowiedź agenta
        """
        # Składamy całą historię debaty w jeden komunikat
        debate_so_far = "\n\n".join(conversation_history)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": debate_so_far + "\n\nTwoja odpowiedź:"},
        ]

        return _generate(self.model, self.tokenizer, messages, config)


class Judge:
    """Sędzia — podsumowuje debatę na końcu."""

    def __init__(self, system_prompt, model, tokenizer):
        self.system_prompt = system_prompt
        self.model = model
        self.tokenizer = tokenizer

    def summarize(self, debate_log, topic, config):
        """Podsumowuje całą debatę.

        Args:
            debate_log: lista dict {agent, round, text}
            topic: temat debaty
            config: dict z parametrami generowania
        Returns:
            str — podsumowanie sędziego
        """
        transcript = f"Temat debaty: {topic}\n\n"
        for entry in debate_log:
            transcript += f"[Runda {entry['round']}] {entry['agent']}: {entry['text']}\n\n"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": transcript + "Podsumuj tę debatę:"},
        ]

        return _generate(self.model, self.tokenizer, messages, config)
