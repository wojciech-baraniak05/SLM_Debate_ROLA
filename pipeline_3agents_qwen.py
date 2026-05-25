# %%
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
# %%
# żeby szybciej się pobierał model, trzeba pobrać z "pip install hf-transfer"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# %%
MODEL_NAME = "Qwen/Qwen3.5-2B"
ARCHITECTURE = 'round_robin'
DECISION_PROTOCOL = 'consensus'

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

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
            {"role": "system",  "content": self.system_prompt},
            {"role": "user",    "content": debate_so_far + "\n\nYour answer:"},
        ]
        return _generate(self.model, self.tokenizer, messages, config)


# ---------------------------------------------------------------------------
# Architecture — round_robin works for any number of agents
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Decision — consensus (works for any number of agents)
# ---------------------------------------------------------------------------

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


def consensus_decision(agents, debate_log, topic, config):
    threshold  = config.get("consensus_threshold", 0.66)
    max_rounds = config.get("max_consensus_rounds", 3)

    print(f"\n{'='*60}")
    print(f"  PROTOCOL: CONSENSUS — threshold: {threshold:.0%}, agents: {len(agents)}")
    print(f"{'='*60}")

    transcript       = _format_transcript(debate_log, topic)
    decision_log     = []
    current_proposal = None
    current_entry    = {}

    for round_num in range(1, max_rounds + 1):
        print(f"\n--- ROUND {round_num}/{max_rounds} ---")

        if current_proposal is None:
            proposer = agents[0]
            messages = [
                {"role": "system", "content": proposer.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{transcript}\n\n"
                        "Formulate a proposal for a common position of all agents "
                        "in one sentence that could reconcile everyone."
                    ),
                },
            ]
            current_proposal = _generate(proposer.model, proposer.tokenizer, messages, config)
            current_entry = {"agent": proposer.name, "proposal": current_proposal}
            print(f"\n[Proposal from {proposer.name}]: {current_proposal}")

        agreements = []
        for agent in agents:
            messages = [
                {"role": "system", "content": agent.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Topic: {topic}\n\n"
                        f"Current proposal:\n\"{current_proposal}\"\n\n"
                        "Do you agree with this proposal? Answer only YES or NO."
                    ),
                },
            ]
            response = _generate(agent.model, agent.tokenizer, messages, config)
            agrees = _parse_yes_no(response)
            agreements.append(agrees)
            current_entry[agent.name] = agrees
            print(f"  [{agent.name}]: {'YES' if agrees else 'NO'}")

        yes_ratio = sum(agreements) / len(agreements)
        decision_log.append(current_entry)

        if yes_ratio >= threshold:
            print(f"\n[CONSENSUS REACHED]: {current_proposal}")
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
                    "Modify the proposal to make it more acceptable to everyone. "
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


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_debate_result(data_dict, config_name: str, base_dir: str = "data"):
    """
    Zapisuje wynik debaty do:
        data/<config_name>/debate-N-result.json

    Każda konfiguracja ma własny podfolder — łatwo oddzielić eksperymenty.
    """
    data_dir = Path(base_dir) / config_name
    data_dir.mkdir(parents=True, exist_ok=True)
    file_count = sum(1 for item in data_dir.iterdir() if item.is_file())
    file_path  = data_dir / f"debate-{file_count + 1}-result.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4)
    print(f"Saved → {file_path}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(config):
    """
    Przeprowadza jedną debatę i zapisuje wynik do:
        data/<config['config_name']>/debate-N-result.json

    config musi zawierać klucz 'config_name'.
    """
    config_name = config["config_name"]

    agents = [
        Agent(
            name=a["name"],
            system_prompt=a["system_prompt"],
            model=model,
            tokenizer=tokenizer,
        )
        for a in config["agents"]
    ]

    print(f"\n[{config_name}]")
    print(f"Architecture:      {ARCHITECTURE}")
    print(f"Decision protocol: {DECISION_PROTOCOL}")
    print(f"Topic:             {config['topic']}")
    print(f"Rounds:            {config['num_rounds']}")
    print(f"Agents ({len(agents)}):      {', '.join(a.name for a in agents)}")

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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# %%
device = 'cpu'
dtype  = torch.float16 if device == "cuda" else torch.float32

print(f"Loading model: {MODEL_NAME} → {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
print("Model ready.\n")


# ===========================================================================
# TOPIC i BASE (wspólne dla wszystkich konfiguracji)
# ===========================================================================

TOPIC = (
    "**PROBLEM**: The company is in financial trouble. "
    "Is it better to lay off 30% of the employees to save the remaining 70%, "
    "or to cut everyone's pay by 20% but not lay anyone off?"
)

BASE = {
    "temperature":          0.7,
    "max_new_tokens":       256,
    "do_sample":            True,
    "num_rounds":           2,
    "topic":                TOPIC,
    "consensus_threshold":  0.66,
    "max_consensus_rounds": 3,
}

# Liczba powtórzeń każdej konfiguracji
N_RUNS = 10


# ===========================================================================
# CONFIGURATION 1 — Proponent, Opponent, Opponent
# Wyniki: data/config_1_proponent_opponent_opponent/
# ===========================================================================

# %% [markdown]
# # Config 1: Proponent + Opponent + Opponent

# %%
config_1 = {
    **BASE,
    "config_name": "config_1_proponent_opponent_opponent",
    "agents": [
        {
            "name": "Proponent",
            "system_prompt": (
                "You are the Proponent in a three-way debate. "
                "You propose solutions and defend them. "
                "There are two Opponents who will challenge you — address both of them. "
                "Be concise (2-3 sentences). Do not repeat arguments you have already made."
            ),
        },
        {
            "name": "Opponent_1",
            "system_prompt": (
                "You are Opponent 1 in a three-way debate. "
                "You challenge the Proponent's arguments and look for flaws. "
                "There is also another Opponent — you may align with them or take a distinct critical angle. "
                "Be concise (2-3 sentences)."
            ),
        },
        {
            "name": "Opponent_2",
            "system_prompt": (
                "You are Opponent 2 in a three-way debate. "
                "You challenge the Proponent's arguments independently of Opponent 1. "
                "Raise different objections if possible, or reinforce Opponent 1's strongest point. "
                "Be concise (2-3 sentences)."
            ),
        },
    ],
}

for _ in range(N_RUNS):
    run(config_1)


# ===========================================================================
# CONFIGURATION 2 — Opponent, Opponent, Proponent
# (te same role, odwrócona kolejność — test order importance)
# Wyniki: data/config_2_opponent_opponent_proponent/
# ===========================================================================

# %% [markdown]
# # Config 2: Opponent + Opponent + Proponent

# %%
config_2 = {
    **BASE,
    "config_name": "config_2_opponent_opponent_proponent",
    "agents": [
        {
            "name": "Opponent_1",
            "system_prompt": (
                "You are Opponent 1 in a three-way debate. "
                "You challenge the Proponent's arguments and look for flaws. "
                "There is also another Opponent — you may align with them or take a distinct critical angle. "
                "Be concise (2-3 sentences)."
            ),
        },
        {
            "name": "Opponent_2",
            "system_prompt": (
                "You are Opponent 2 in a three-way debate. "
                "You challenge the Proponent's arguments independently of Opponent 1. "
                "Raise different objections if possible, or reinforce Opponent 1's strongest point. "
                "Be concise (2-3 sentences)."
            ),
        },
        {
            "name": "Proponent",
            "system_prompt": (
                "You are the Proponent in a three-way debate. "
                "You propose solutions and defend them. "
                "There are two Opponents who will challenge you — address both of them. "
                "Be concise (2-3 sentences). Do not repeat arguments you have already made."
            ),
        },
    ],
}

for _ in range(N_RUNS):
    run(config_2)


# ===========================================================================
# CONFIGURATION 3 — Employee, Employee, Boss
# Wyniki: data/config_3_employee_employee_boss/
# ===========================================================================

# %% [markdown]
# # Config 3: Employee + Employee + Boss

# %%
config_3 = {
    **BASE,
    "config_name": "config_3_employee_employee_boss",
    "agents": [
        {
            "name": "Employee_1",
            "system_prompt": (
                "You are Employee 1. You report to the Boss. "
                "You represent the workers' perspective — job security, morale, fairness. "
                "There is another employee alongside you; you may support each other's arguments. "
                "Be concise (2-3 sentences)."
            ),
        },
        {
            "name": "Employee_2",
            "system_prompt": (
                "You are Employee 2. You report to the Boss. "
                "You represent the workers' perspective — job security, morale, fairness. "
                "There is another employee alongside you; add new arguments where possible. "
                "Be concise (2-3 sentences)."
            ),
        },
        {
            "name": "Boss",
            "system_prompt": (
                "You are the Boss. Your decisions carry the most weight in the company. "
                "You must consider financial sustainability alongside employee welfare. "
                "Two employees will share their perspectives — listen and respond decisively. "
                "Be concise (2-3 sentences)."
            ),
        },
    ],
}

for _ in range(N_RUNS):
    run(config_3)


# ===========================================================================
# CONFIGURATION 4 — Boss, Boss, Employee
# Wyniki: data/config_4_boss_boss_employee/
# ===========================================================================

# %% [markdown]
# # Config 4: Boss + Boss + Employee

# %%
config_4 = {
    **BASE,
    "config_name": "config_4_boss_boss_employee",
    "agents": [
        {
            "name": "Boss_1",
            "system_prompt": (
                "You are Boss 1, a senior manager focused on financial survival. "
                "Your opinion carries significant weight. "
                "Boss 2 is your peer — you may agree or propose a different managerial approach. "
                "Be concise (2-3 sentences)."
            ),
        },
        {
            "name": "Boss_2",
            "system_prompt": (
                "You are Boss 2, a senior manager who values team cohesion. "
                "Your opinion carries significant weight. "
                "Boss 1 is your peer — engage with their arguments and offer your own perspective. "
                "Be concise (2-3 sentences)."
            ),
        },
        {
            "name": "Employee",
            "system_prompt": (
                "You are an Employee. You report to both bosses. "
                "You represent the workers' perspective — job security, morale, fairness. "
                "Two bosses are debating — add the employee viewpoint they may be missing. "
                "Be concise (2-3 sentences)."
            ),
        },
    ],
}

for _ in range(N_RUNS):
    run(config_4)


# ===========================================================================
# CONFIGURATION 5 — Student, Student, Teacher
# Wyniki: data/config_5_student_student_teacher/
# ===========================================================================

# %% [markdown]
# # Config 5: Student + Student + Teacher

# %%
config_5 = {
    **BASE,
    "config_name": "config_5_student_student_teacher",
    "agents": [
        {
            "name": "Student_1",
            "system_prompt": (
                "You are Student 1. You are discussing a business ethics problem with a Teacher and another Student. "
                "You reason from first principles — fairness, human impact, social consequences. "
                "Be open to learning but stand by your reasoning. "
                "Be concise (2-3 sentences)."
            ),
        },
        {
            "name": "Student_2",
            "system_prompt": (
                "You are Student 2. You are discussing a business ethics problem with a Teacher and another Student. "
                "You may agree or respectfully disagree with Student 1 — bring your own angle. "
                "Be open to learning but stand by your reasoning. "
                "Be concise (2-3 sentences)."
            ),
        },
        {
            "name": "Teacher",
            "system_prompt": (
                "You are the Teacher. You guide the discussion between two Students. "
                "You provide expertise, correct misconceptions, and help them reach a well-reasoned answer. "
                "Your role is authoritative but pedagogical — explain your reasoning. "
                "Be concise (2-3 sentences)."
            ),
        },
    ],
}

for _ in range(N_RUNS):
    run(config_5)


# ===========================================================================
# CONFIGURATION 6 — Teacher, Teacher, Student
# Wyniki: data/config_6_teacher_teacher_student/
# ===========================================================================

# %% [markdown]
# # Config 6: Teacher + Teacher + Student

# %%
config_6 = {
    **BASE,
    "config_name": "config_6_teacher_teacher_student",
    "agents": [
        {
            "name": "Teacher_1",
            "system_prompt": (
                "You are Teacher 1, an expert focused on economic and organizational reasoning. "
                "You are guiding a discussion with Teacher 2 and a Student. "
                "Engage with Teacher 2's perspective and model good argumentation for the Student. "
                "Be concise (2-3 sentences)."
            ),
        },
        {
            "name": "Teacher_2",
            "system_prompt": (
                "You are Teacher 2, an expert focused on ethical and human-centered reasoning. "
                "You are guiding a discussion with Teacher 1 and a Student. "
                "Engage with Teacher 1's perspective and model good argumentation for the Student. "
                "Be concise (2-3 sentences)."
            ),
        },
        {
            "name": "Student",
            "system_prompt": (
                "You are the Student. Two Teachers are guiding the discussion. "
                "You reason from first principles — fairness, human impact, social consequences. "
                "Engage with both teachers' arguments and develop your own position. "
                "Be concise (2-3 sentences)."
            ),
        },
    ],
}

for _ in range(N_RUNS):
    run(config_6)
