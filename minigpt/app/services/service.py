from minigpt.models.minigpt import MiniGPT

model = MiniGPT.load_from_checkpoint(
    checkpoint_path="checkpoints/model.ckpt",
    map_location="cpu"
)

def generate_text(prompt: str, max_length: int) -> str:
    return model.generate_text(input_text=prompt, context_size=256, num_tokens_to_generate=max_length)