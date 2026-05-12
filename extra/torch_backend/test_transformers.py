import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationMixin
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
import unittest
import tinygrad.nn.torch  # noqa
from tinygrad.helpers import getenv
from tinygrad import Device
import torch


if getenv("TINY_BACKEND"):
  device = torch.device("tiny")
else:
  device = torch.device({"METAL": "mps", "NV": "cuda"}.get(Device.DEFAULT, "cpu"))


class TestTorchBackendInplace(unittest.TestCase):
  def test_model_to_tiny(self):
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    messages = [
      {"role": "user", "content": "Who are you?"},
    ]
    text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
      enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))


if __name__ == "__main__":
  unittest.main(verbosity=2)
