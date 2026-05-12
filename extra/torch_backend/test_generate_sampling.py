from tinygrad import Tensor
import tinygrad.nn.torch  # noqa
from tinygrad.helpers import getenv
from tinygrad import Device
from torch import nn
import torch
import unittest

if getenv("TINY_BACKEND"):
  device = torch.device("tiny")
else:
  device = torch.device({"METAL": "mps", "NV": "cuda"}.get(Device.DEFAULT, "cpu"))


class DummyGPT(nn.Module):
  def __init__(self, dim: int, vocab: int):
    super().__init__()
    self.dim = dim
    self.vocab = vocab
    self.emb_w = nn.Embedding(vocab, dim)
    self.qkv = nn.Linear(dim, dim * 3)
    self.proj = nn.Linear(dim, dim)
    self.ln = nn.LayerNorm(dim)
    self.ff_up = nn.Linear(dim, dim * 4)
    self.ff_dn = nn.Linear(dim * 4, dim)
    self.head = nn.Linear(dim, vocab)

  def forward(self, ids: torch.Tensor) -> torch.Tensor:
    B, T = ids.shape
    h = self.emb_w(ids)
    r = h
    h = self.ln(h)
    qkv = self.qkv(h)
    q, k, v = qkv.split(qkv.shape[-1] // 3, dim=-1)
    s = q.shape[-1] ** 0.5
    att = ((q @ k.transpose(-1, -2)) / s).softmax(-1)
    h = r + self.proj(att @ v)
    r = h
    h = (h - h.mean(-1, keepdim=True)) * ((h.var(-1, keepdim=True) + 1e-5) ** -0.5)
    h = r + nn.functional.gelu(self.ff_dn(self.ff_up(h)))
    return self.head(h)


def generation_loop(model, input_ids, max_new, temperature, do_sample):
  for _ in range(max_new):
    logits = model.forward(input_ids)
    next_logits = logits[:, -1, :] / temperature
    if do_sample:
      probs = next_logits.softmax(-1)
      next_token = probs.multinomial(num_samples=1)
    else:
      next_token = next_logits.argmax(-1, keepdim=True)
    input_ids = torch.cat([input_ids, next_token], dim=-1)
  return input_ids


def generation_loop(model, input_ids, max_new, temperature, do_sample):
  for _ in range(max_new):
    logits = model.forward(input_ids)
    next_logits = logits[:, -1, :] / temperature
    if do_sample:
      probs = next_logits.softmax(-1)
      next_token = probs.multinomial(num_samples=1)
    else:
      next_token = next_logits.argmax(-1, keepdim=True)
    input_ids = torch.cat([input_ids, next_token], dim=-1)
  return input_ids


class TestGenerateSamplingBug(unittest.TestCase):
  def test_generation_loop(self):
    model = DummyGPT(128, 4096).to(device)
    input_ids = torch.tensor([[1, 2, 3]]).to(device)
    output = generation_loop(model, input_ids, max_new=10, temperature=1.0, do_sample=True)
    print(output)
    self.assertEqual(output.shape, (1, 13))


if __name__ == "__main__":
  unittest.main(verbosity=2)
