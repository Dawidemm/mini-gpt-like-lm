import torch

from typing import Dict, Tuple, List, Union


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

def collate_func(
        batch: List[Union[List[int], torch.Tensor]],
        padding_token_id: int = 50256,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

    batch_max_length = max(len(item)+1 for item in batch)
    inputs_list, targets_list = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [padding_token_id]

        padded = (
            new_item + [padding_token_id] * (batch_max_length - len(new_item))
        )

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == padding_token_id
        indices = torch.nonzero(mask).squeeze()

        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        inputs_list.append(inputs)
        targets_list.append(targets)

    inputs_tensor = torch.stack(inputs_list)
    targets_tensor = torch.stack(targets_list)

    return inputs_tensor, targets_tensor