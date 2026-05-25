# %%
import json
import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# żeby szybciej się pobierał model, trzeba pobrać z "pip install hf-transfer"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# %%
MODEL_NAME = "Qwen/Qwen3.5-2B"
ARCHITECTURE = 'round_robin'
DECISION_PROTOCOL = 'consensus'

# %%
# Agent
def _generate(model, tokenizer, messages, config):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.get("max_new_tokens", 256),
            max_length=None,
            temperature=config.get("temperature", 0.7),
            do_sample=config.get("do_sample", True),
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class Agent:
    def __init__(self, name, system_prompt, model, tokenizer):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.tokenizer = tokenizer

    def respond(self, conversation_history, config):
        debate_so_far = "\n\n".join(conversation_history)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": debate_so_far + "\n\nYour answer:"},
        ]
        return _generate(self.model, self.tokenizer, messages, config)


# Architecture
def round_robin(agents, topic, num_rounds, config):
    debate_log = []
    history = [f"Debate topic: {topic}"]

    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"  ROUND {round_num}")
        print(f"{'='*60}")

        for agent in agents:
            response = agent.respond(history, config)
            debate_log.append({"agent": agent.name, "round": round_num, "text": response})
            history.append(f"{agent.name}: {response}")
            print(f"\n[{agent.name}]: {response}")

    return debate_log


# Decision
def consensus_decision(agents, debate_log, topic, config):
    threshold  = config.get("consensus_threshold", 0.66)
    max_rounds = config.get("max_consensus_rounds", 3)

    print(f"\n{'='*60}")
    print(f"  PROTOCOL: CONSENSUS — threshold: {threshold:.0%}")
    print(f"{'='*60}")

    transcript       = _format_transcript(debate_log, topic)
    decision_log     = []
    current_proposal = None
    current_entry    = {}

    for round_num in range(1, max_rounds + 1):
        print(f"\n--- ROUND {round_num}/{max_rounds} ---")

        if current_proposal is None:
            messages = [
                {"role": "system", "content": agents[0].system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{transcript}\n\n"
                        "Formulate a proposal for a common position of all agents "
                        "in one sentence that could reconcile them."
                    ),
                },
            ]
            current_proposal = _generate(agents[0].model, agents[0].tokenizer, messages, config)
            current_entry = {"agent": agents[0].name, "proposal": current_proposal}
            print(f"\n[Proposal from {agents[0].name}]: {current_proposal}")

        agreements = []
        for agent in agents:
            messages = [
                {"role": "system", "content": agent.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Topic: {topic}\n\n"
                        f"Current proposal:\n\"{current_proposal}\"\n\n"
                        "Do you agree with current proposal? Answer only YES or NO."
                    ),
                },
            ]
            response = _generate(agent.model, agent.tokenizer, messages, config)
            agrees = _parse_yes_no(response)
            agreements.append(agrees)
            current_entry[agent.name] = agrees
            print(f"  [{agent.name}]: {'YES' if agrees else 'NO'}")

        n         = len(agreements)
        yes_ratio = sum(agreements) / n
        no_ratio  = (n - sum(agreements)) / n
        decision_log.append(current_entry)

        if yes_ratio >= threshold or no_ratio >= threshold:
            dominant = "YES" if yes_ratio >= no_ratio else "NO"
            print(f"\n[CONSENSUS — majority {dominant}]:\n{current_proposal}")
            return decision_log, current_proposal

        if round_num == max_rounds:
            break

        print("  --- NO CONSENSUS YET, MODIFYING PROPOSAL ---")
        dissenters = [a for a, ok in zip(agents, agreements) if not ok]
        modifier   = dissenters[0] if dissenters else agents[-1]
        messages = [
            {"role": "system", "content": modifier.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\n\n"
                    f"Current proposal:\n\"{current_proposal}\"\n\n"
                    "Modify current proposal to make it more acceptable to everyone. "
                    "Answer in one sentence."
                ),
            },
        ]
        current_proposal = _generate(modifier.model, modifier.tokenizer, messages, config)
        current_entry    = {"agent": modifier.name, "proposal": current_proposal}
        print(f"  [{modifier.name} proposes]: {current_proposal}")

    print(f"\n[NO CONSENSUS after {max_rounds} rounds — returning last proposal]:")
    print(current_proposal)
    return decision_log, current_proposal


def _parse_yes_no(text):
    t = text.strip().lower()
    if t.startswith("tak") or t.startswith("yes"):
        return True
    if t.startswith("nie") or t.startswith("no"):
        return False
    head = t[:30]
    if "tak" in head or "yes" in head:
        return True
    return False


def _format_transcript(debate_log, topic):
    out = f"Topic: {topic}\n\n"
    for entry in debate_log:
        out += f"[Round {entry['round']}] {entry['agent']}: {entry['text']}\n\n"
    return out


def save_debate_result(data_dict, config_name: str, base_dir: str = "data"):
    data_dir   = Path(base_dir) / config_name
    data_dir.mkdir(parents=True, exist_ok=True)
    file_count = sum(1 for item in data_dir.iterdir() if item.is_file())
    file_path  = data_dir / f"debate-{file_count + 1}-result.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4)
    print(f"Successfully saved {file_path}")


# %%
def run(config):
    config_name = config["config_name"]

    agents = [
        Agent(name=a["name"], system_prompt=a["system_prompt"], model=model, tokenizer=tokenizer)
        for a in config["agents"]
    ]

    print(f"\n[{config_name}]")
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Decision protocol: {DECISION_PROTOCOL}")
    print(f"Topic: {config['topic']}")
    print(f"Rounds: {config['num_rounds']}")
    print(f"Agents: {', '.join(a.name for a in agents)}")

    debate_log = round_robin(agents, config["topic"], config["num_rounds"], config)
    decision_log, final_answer = consensus_decision(agents, debate_log, config["topic"], config)

    save_debate_result(
        data_dict={
            "config_name":         config_name,
            "model":               MODEL_NAME,
            "architecture":        ARCHITECTURE,
            "decision_protocol":   DECISION_PROTOCOL,
            "temperature":         config.get("temperature", 0.7),
            "max_new_tokens":      config.get("max_new_tokens", 256),
            "do_sample":           config.get("do_sample", True),
            "num_rounds":          config.get("num_rounds", 2),
            "topic":               config.get("topic", ""),
            "consensus_threshold": config.get("consensus_threshold", 0.66),
            "max_consensus_rounds":config.get("max_consensus_rounds", 3),
            "agents": [
                {"name": a["name"], "system_prompt": a["system_prompt"]}
                for a in config["agents"]
            ],
            "debate_log":   debate_log,
            "decision_log": decision_log,
        },
        config_name=config_name,
    )

    print(f"\n{'='*60}")
    print(f"  FINAL VERDICT (protocol: {DECISION_PROTOCOL})")
    print(f"{'='*60}")
    print(f"\n{final_answer}\n")


# %%
# Model loading
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype  = torch.float16 if device == "cuda" else torch.float32

print(f"Model loading: {MODEL_NAME} to {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
print("Model loaded.\n")

TOPIC = (
    "**PROBLEM**: The company is in financial trouble. "
    "Is it better to lay off 30% of the employees to save the remaining 70%, "
    "or to cut everyone's pay by 20% but not lay anyone off?"
)

BASE = {
    "temperature":          0.5,
    "max_new_tokens":       256,
    "do_sample":            True,
    "num_rounds":           2,
    "topic":                TOPIC,
    "consensus_threshold":  0.66,
    "max_consensus_rounds": 3,
}

N_RUNS = 10


# %% [markdown]
# # Proponent vs Opponent

# %%
config_proponent_opponent = {
    **BASE,
    "config_name": "config_2a_proponent_opponent",
    "agents": [
        {
            "name": "Proponent",
            "system_prompt": (
                "You are an AI agent. You are the Proponent. "
                "Your task is to propose solutions and argue. "
                "The Opponent refutes your arguments. "
                "Defend your arguments, but look for a solution that suits both sides. "
                "You are conducting a conversation and seeking a common answer. "
                "Do not repeat your arguments. "
                "Your goal is to find the best solution to the given PROBLEM as quickly as possible. "
                "Answer concisely in 2-3 sentences."
            ),
        },
        {
            "name": "Opponent",
            "system_prompt": (
                "You are an AI agent. You are the Opponent. "
                "Take a position different from the Proponent. "
                "Your task is to respond to the Proponent's arguments. "
                "You criticize, but you seek a solution that suits both sides. "
                "You are conducting a conversation and looking for an answer. "
                "Your goal is to find the best solution to the given PROBLEM as quickly as possible. "
                "Answer concisely in 2-3 sentences."
            ),
        },
    ],
}

for _ in range(N_RUNS):
    run(config_proponent_opponent)


# %% [markdown]
# # Employee vs Boss

# %%
config_employee_boss = {
    **BASE,
    "config_name": "config_2a_employee_boss",
    "agents": [
        {
            "name": "Employee",
            "system_prompt": (
                "You are an Employee in the company. "
                "Boss is your direct superior. "
                "He assigns you tasks and evaluates your performance. "
                "You are conducting a conversation and looking for a common answer. "
                "Your goal is to find the best solution to the PROBLEM together as quickly as possible. "
                "Answer concisely in 2-3 sentences."
            ),
        },
        {
            "name": "Boss",
            "system_prompt": (
                "You are the Boss of the company. "
                "The Employee follows your orders. "
                "Your opinion counts more in the company. "
                "You lead the conversation and look for answers. "
                "Your goal is to find the best solution to the given PROBLEM as quickly as possible. "
                "Answer concisely in 2-3 sentences."
            ),
        },
    ],
}

for _ in range(N_RUNS):
    run(config_employee_boss)


# %% [markdown]
# # Proponent vs Opponent — stronger prompts

# %%
config_proponent_opponent_strong = {
    **BASE,
    "temperature":  0.7,
    "config_name": "config_2a_proponent_opponent_strong",
    "agents": [
        {
            "name": "Proponent",
            "system_prompt": (
                "You are the Proponent. "
                "You propose solutions and argue. "
                "The Opponent refutes your arguments. "
                "Defend your arguments. "
                "You are conducting a conversation. "
                "Do not repeat your arguments. Do not moderate. "
                "Your goal is to find the best solution to the given PROBLEM as quickly as possible. "
                "Answer concisely in 2-3 sentences."
            ),
        },
        {
            "name": "Opponent",
            "system_prompt": (
                "You are the Opponent. "
                "You refute the Proponent's arguments. "
                "You do not agree with the Proponent. "
                "You take a position different from the Proponent. "
                "You counterargument and criticize their arguments. "
                "You are conducting a conversation. "
                "Your goal is to find the best solution to the given PROBLEM as quickly as possible. "
                "Answer concisely in 2-3 sentences."
            ),
        },
    ],
}

for _ in range(N_RUNS):
    run(config_proponent_opponent_strong)


# %% [markdown]
# # Proponent vs Opponent — neutral prompts

# %%
config_proponent_opponent_neutral = {
    **BASE,
    "config_name": "config_2a_proponent_opponent_neutral",
    "agents": [
        {
            "name": "Proponent",
            "system_prompt": (
                "You are a proponent. "
                "You propose solutions to a given problem. "
                "You talk with an opponent. "
                "You are looking for a common solution. "
                "Answer concisely in 2-3 sentences."
            ),
        },
        {
            "name": "Opponent",
            "system_prompt": (
                "You are an opponent. "
                "You look for gaps in the proposed solutions. "
                "You talk with the proponent. "
                "You are looking for a common solution. "
                "Answer concisely in 2-3 sentences."
            ),
        },
    ],
}

for _ in range(N_RUNS):
    run(config_proponent_opponent_neutral)
