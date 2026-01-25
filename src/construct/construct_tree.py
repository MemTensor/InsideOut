import json
from pathlib import Path

# Adjust these imports to match your actual file/module names
from .attribute_extraction import Prompt        # contains class Prompt
from src.llm import Agent         # contains class Agent
from .attribute_extraction import apply_ops_to_tree # contains apply_ops_to_tree()


def load_dialogue_from_json(input_dialogue) -> str:
    """
    Load dialogue text from input_dialogue.json.

    This is made tolerant to a few common JSON shapes:
    1) A plain string:
       "some dialogue text..."

    2) A dict with a dialogue-like field:
       { "dialogue": "..." }
       { "text": "..." }
       { "content": "..." }
       { "input": "..." }

    3) A list of messages:
       [
         {"role": "user", "content": "..."},
         {"role": "assistant", "content": "..."},
         ...
       ]
       In this case we concatenate them into a single text.
    """
    data = input_dialogue
    # Case 1: plain string
    if isinstance(data, str):
        return data

    # Case 2: single dict with common keys
    if isinstance(data, dict):
        for key in ["dialogue", "text", "content", "input"]:
            if key in data and isinstance(data[key], str):
                return data[key]
        # Fallback: pretty-print the dict as JSON
        return json.dumps(data, ensure_ascii=False, indent=2)

    # Case 3: list (e.g., list of messages)
    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, dict) and "content" in item:
                role = item.get("role", "")
                content = item["content"]
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(item))
        return "\n\n".join(parts)

    # Fallback: convert unknown structure to string
    return str(data)


def run_full_pipeline(
    input_dialogue,
    memtree: str,
    attribute_temp: float = 0.3,
    ops_temp: float = 0.2,
    max_tokens: int = 24576,
):
    """
    Full pipeline:
    1) Load dialogue from JSON.
    2) Step 1: dialogue -> attribute paragraph via LLM.
    3) Step 2: attribute paragraph -> operations via LLM.
    4) Step 3: apply operations to persona schema.
    5) Save updated tree to updated_tree.json.
    """

    # ---------- Step 0: load dialogue ----------
    dialogue_text = load_dialogue_from_json(input_dialogue)
    print("=== Loaded dialogue ===")
    # print(dialogue_text)
    # print()

    # ---------- Step 1: dialogue -> attribute paragraph ----------
    prompt=Prompt(memtree)
    if len(input_dialogue)==1 and input_dialogue[0]["role"]=="system":
        step1_prompt = prompt.attribute_extraction_system(dialogue_text)
    else:
        step1_prompt = prompt.attribute_extraction(dialogue_text)

    attr_agent = Agent(
        system_prompt="You extract stable personal attributes from dialogue and return a single descriptive paragraph."
    )

    attribute_text, ok1 = attr_agent.run(
        prompt=step1_prompt,
        temperature=attribute_temp,
        top_p=0.8,
        max_length=max_tokens,
    )

    print("=== Step 1: attribute_text ===")
    print(attribute_text)
    print("success:", ok1)
    print()

    # ---------- Step 2: attribute paragraph -> operations ----------
    step2_prompt = prompt.attribute_to_tree_ops(attribute_text)

    ops_agent = Agent(
        system_prompt="You convert attribute paragraphs into tree operations. "
                     "You must output ONLY ADD/UPDATE/DELETE/NO_OP lines."
    )

    ops_text, ok2 = ops_agent.run(
        prompt=step2_prompt,
        temperature=ops_temp,
        top_p=0.9,
        max_length=max_tokens,
    )

    print("=== Step 2: operations ===")
    print(ops_text)
    print("success:", ok2)
    print()

    # Path("tree_operations.txt").write_text(ops_text, encoding="utf-8")

    # ---------- Step 3: apply operations to schema ----------
    updated_tree = apply_ops_to_tree(
        base_schema_json=prompt.PERSONA_SCHEMA_JSON,
        ops_text=ops_text,
        delete_sets_empty=True,
    )

    print("=== Step 3: updated_tree saved to updated_tree.json ===")

    return {
        "dialogue": dialogue_text,
        "attribute_text": attribute_text,
        "operations": ops_text,
        "updated_tree": updated_tree,
        "step1_success": ok1,
        "step2_success": ok2,
    }


def run_full_pipeline_1(
    input_dialogue,
    memtree: str,
    ops_temp: float = 0.2,
    max_tokens: int = 24576,
):
    """
    Full pipeline:
    1) Load dialogue from JSON.
    2) Step 1: dialogue -> attribute paragraph via LLM.
    3) Step 2: attribute paragraph -> operations via LLM.
    4) Step 3: apply operations to persona schema.
    5) Save updated tree to updated_tree.json.
    """

    # ---------- Step 0: load dialogue ----------
    dialogue_text = load_dialogue_from_json(input_dialogue)
    print("=== Loaded dialogue ===")
    # print(dialogue_text)
    # print()

    # ---------- Step 1: dialogue -> attribute paragraph ----------
    prompt=Prompt(memtree)

    # Optionally save intermediate result
    # Path("attribute_text.txt").write_text(attribute_text, encoding="utf-8")

    # ---------- Step 2: attribute paragraph -> operations ----------
    step_prompt = prompt.dialogue_to_tree_ops(dialogue_text)

    ops_agent = Agent(
        system_prompt="You convert dialogue into tree operations. "
                     "You must output ONLY ADD/UPDATE/DELETE/NO_OP lines."
    )

    ops_text, ok = ops_agent.run(
        prompt=step_prompt,
        temperature=ops_temp,
        top_p=0.9,
        max_length=max_tokens,
    )

    print("=== Step 2: operations ===")
    print(ops_text)
    print("success:", ok)
    print()

    # Path("tree_operations.txt").write_text(ops_text, encoding="utf-8")

    # ---------- Step 3: apply operations to schema ----------
    updated_tree = apply_ops_to_tree(
        base_schema_json=prompt.PERSONA_SCHEMA_JSON,
        ops_text=ops_text,
        delete_sets_empty=True,
    )

    print("=== Step 3: updated_tree saved to updated_tree.json ===")
    
    return {
        "dialogue": dialogue_text,
        "operations": ops_text,
        "previous_tree": memtree,
        "updated_tree": updated_tree,
        "step_success": ok,
    }




if __name__ == "__main__":
    with open("src/construct/input_dialogue.json", "r", encoding="utf-8") as f:
        input_dialogue = json.load(f)
    run_full_pipeline(input_dialogue)
