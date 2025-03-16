from minigpt.models.minigpt import MiniGPT

model = MiniGPT.load_from_checkpoint(
    checkpoint_path="checkpoints/model.ckpt",
    map_location="cpu"
)

def generate(prompt: str, max_length: int) -> str:
    return model.generate(prompt=prompt)