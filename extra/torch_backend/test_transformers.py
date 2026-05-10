import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import unittest
import tinygrad.nn.torch # noqa

class TestTorchBackendInplace(unittest.TestCase):
    def test_model_to_tiny(self):
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").tiny()
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
        outputs = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))


if __name__ == "__main__":
  unittest.main(verbosity=2)
