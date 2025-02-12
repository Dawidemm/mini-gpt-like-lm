import torch

from typing import Dict


def format_input_text(
        raw_txt: Dict
    ):

    instruction_text = (
        f"Below in an instruction that describes a task."
        f"Write a response that appropriately completes the request. "
        f"\n\n### Instruction: \n{raw_txt['instruction']}"
    )

    input_text = f"\n\n### Input: \n{raw_txt['input']}" if raw_txt['input'] else ""

    response_text = f"\n\n### Response: \n{raw_txt['response']}"

    return instruction_text + input_text + response_text