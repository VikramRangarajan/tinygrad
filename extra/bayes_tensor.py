"""
BayesTensor: an OpMixin subclass that propagates both E[X] and Var[X] through a graph.

Only ONE primitive needs moment rules: `alu` (elementwise). Everything else (matmul,
dot, sum, reshape, transpose, ...) is inherited for free from OpMixin, because those ops
are each defined exactly once in terms of alu / _rop / _mop.

The point of this file is that gradients work in BOTH senses:

  * E[gradient]  -- standard tinygrad autodiff on the MEAN graph (loss.mean.backward())
                    gives the expected gradient dE[L]/d(theta). That is the quantity you
                    would feed an SGD step on the posterior means.

  * Var[gradient] -- the gradient w.r.t. a random parameter is itself a random quantity
                    (it is a function of the other random values in the net). We take the
                    gradient UOp graph straight out of compute_gradient, rebuild it as a
                    BayesTensor by pairing each forward tensor with its own variance, and
                    propagate moments through it. This is exactly differentiating the
                    random loss L directly, no extra derivation needed.

Both are verified against closed form AND against Monte Carlo.
"""

from __future__ import annotations
import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import argfix
from tinygrad.mixin.op import OpMixin
from tinygrad.uop import Ops, GroupOp
from tinygrad.uop.ops import UOp
from tinygrad.mixin.gradient import compute_gradient


class BayesTensor(OpMixin):
  __slots__ = 'mean', 'var'

  def __init__(self, mean: Tensor, var: Tensor): self.mean, self.var = mean, var

  # ---- the ~8-method abstract surface OpMixin needs ----
  @property
  def shape(self): return self.mean.shape
  @property
  def dtype(self): return self.mean.dtype
  @property
  def device(self): return self.mean.device

  @classmethod
  def const(cls, dtype, b): return cls(Tensor.const(dtype, b), Tensor.const(dtype, 0))

  @classmethod
  def _wrap_uop(cls, u): raise NotImplementedError("cast/ufix are overridden so this is never hit")

  @property
  def _uop(self): return self.mean.uop

  # scalar / uop lifting -> deterministic quantity with zero variance
  def ufix(self, x):
    if isinstance(x, BayesTensor): return x
    if isinstance(x, UOp): x = self.mean._wrap_uop(x)
    return BayesTensor(self.mean.ufix(x), self.var.ufix(x) * 0)

  def cast(self, dtype):
    if self.mean.dtype == dtype and self.var.dtype == dtype: return self
    return BayesTensor(self.mean.cast(dtype), self.var.cast(dtype))

  # ---- elementwise primitive: the moment propagation rules ----
  def alu(self, op: Ops, *src: BayesTensor) -> BayesTensor:
    if op is Ops.ADD:
      # Var[a+b] = Va + Vb (independence)
      return BayesTensor(self.mean + src[0].mean, self.var + src[0].var)
    if op is Ops.SUB:
      return BayesTensor(self.mean - src[0].mean, self.var + src[0].var)
    if op is Ops.MUL:
      # E[ab] = ma*mb, Var[ab] = ma^2*Vb + mb^2*Va + Va*Vb
      ma, va, mb, vb = self.mean, self.var, src[0].mean, src[0].var
      return BayesTensor(ma * mb, ma * ma * vb + mb * mb * va + va * vb)
    if op is Ops.RECIPROCAL:
      # first-order Taylor: E[1/x] ~ 1/m, Var[1/x] ~ v/m^4
      m, v = self.mean, self.var
      return BayesTensor(m.reciprocal(), v * m.reciprocal() ** 4)
    if op is Ops.DETACH:
      # routing (no derivation): detach both moments from autograd
      return BayesTensor(self.mean.detach(), self.var.detach())
    if op is Ops.CONTIGUOUS_BACKWARD:
      # routing: identity in the backward pass
      return self
    raise NotImplementedError(f"alu rule missing for {op}")

  # ---- reduce primitive: Var[sum] = sum(Var) ----
  def _rop(self, op: Ops, axis: tuple) -> BayesTensor:
    if op is Ops.ADD:
      return BayesTensor(self.mean.sum(axis), self.var.sum(axis))
    raise NotImplementedError(f"_rop rule missing for {op}")

  # ---- movement primitive: apply to both moments ----
  def _mop(self, op: Ops, arg) -> BayesTensor:
    return BayesTensor(self.mean._mop(op, arg), self.var._mop(op, arg))

  # STACK can't go through _mop (its arg is the *uops* of the other tensors,
  # so the vars would be lost) -- override the high-level method instead
  def stack(self, *args, dim=0):
    tensors = argfix(self, *args)
    return BayesTensor(tensors[0].mean.stack(*[t.mean for t in tensors[1:]], dim=dim),
                       tensors[0].var.stack(*[t.var for t in tensors[1:]], dim=dim))

  # contiguous is a no-op semantically for the (mean, var) pair
  def contiguous(self, **kwargs):
    return BayesTensor(self.mean.contiguous(**kwargs), self.var.contiguous(**kwargs))

  def __repr__(self): return f"BayesTensor(mean={self.mean.shape}, var={self.var.shape})"


# ---------------------------------------------------------------------------
# moments(): rebuild an arbitrary UOp expression as a BayesTensor, pairing each
# forward tensor with its variance. Used to propagate E/Var through the
# *gradient graph* -- i.e. to get E[grad] and Var[grad].
# ---------------------------------------------------------------------------
def moments(u: UOp, bt_map: dict[UOp, BayesTensor]) -> BayesTensor:
  if u in bt_map: return bt_map[u]
  if u.op is Ops.CONST:
    return BayesTensor(Tensor.const(u.dtype, u.arg), Tensor.const(u.dtype, 0))
  if u.op is Ops.CAST:
    m = moments(u.src[0], bt_map)
    return BayesTensor(m.mean.cast(u.dtype), m.var.cast(u.dtype))
  if u.op in GroupOp.Movement:
    # EXPAND from a broadcast carries arg=None; the target shape lives on the node
    arg = u.shape if (u.op is Ops.EXPAND and u.arg is None) else u.arg
    return moments(u.src[0], bt_map)._mop(u.op, arg)
  if u.op is Ops.REDUCE:
    return moments(u.src[0], bt_map)._rop(u.arg[0], u.arg[1])
  if u.op in {Ops.ADD, Ops.SUB, Ops.MUL}:
    m = [moments(s, bt_map) for s in u.src]
    return m[0].alu(u.op, *m[1:])
  # a gradient written into a buffer: AFTER(BUFFER, STORE(..., value)) -> the stored value.
  # STORE, COPY and CONTIGUOUS are value-identity wrappers for moment propagation.
  if u.op in {Ops.AFTER, Ops.STORE}: return moments(u.src[1], bt_map)
  if u.op is Ops.COPY: return moments(u.src[0], bt_map)
  if u.op is Ops.CONTIGUOUS: return moments(u.src[0], bt_map).contiguous()
  raise NotImplementedError(f"moments: unhandled {u.op}")


def check(name, got: np.ndarray, want: np.ndarray, tol=1e-4):
  ok = np.allclose(got, want, atol=tol, rtol=tol)
  print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
  assert ok, f"{name}:\n got {got}\nwant {want}"


# ---------------------------------------------------------------------------
# Section 1: forward moments. matmul = (a*b).sum() defined ONCE in OpMixin.
# ---------------------------------------------------------------------------
print("== 1. forward variance propagation ==")
Tensor.manual_seed(42)
a = BayesTensor(Tensor.rand(2, 3), Tensor.rand(2, 3))
b = BayesTensor(Tensor.rand(3, 4), Tensor.rand(3, 4))
c = a @ b
assert c.mean.shape == (2, 4) and c.var.shape == (2, 4)

# closed form: Var[A@B] = sum_k (ma_ik^2 Vb_kj + mb_kj^2 Va_ik + Va_ik Vb_kj)
m = (a.mean * a.mean) @ b.var + a.var @ (b.mean * b.mean) + a.var @ b.var
check("matmul var == closed form", c.var.numpy(), m.numpy())

r = c.reciprocal()  # nonlinear op, Taylor rule
print("  reciprocal E:", r.mean.shape, "Var:", r.var.shape, "(Taylor approx)")

s = a.stack(a2 := BayesTensor(Tensor.rand(2, 3), Tensor.rand(2, 3)), dim=0)  # stack + routing
check("stack -> mean+var stacked", s.mean.numpy(), np.stack([a.mean.numpy(), a2.mean.numpy()]))
assert (s.var.numpy() >= 0).all()
d = c.detach()
check("detach is a no-op", d.mean.numpy(), c.mean.numpy())

# ---------------------------------------------------------------------------
# Section 2: E[gradient] -- tinygrad autodiff on the MEAN graph.
# L = (a*b)*c, so dE[L]/dmu_a = mu_b * mu_c.
# ---------------------------------------------------------------------------
print("\n== 2. expected gradient via loss.mean.backward() ==")
N = 4
rng = np.random.default_rng(0)
ma, va = rng.uniform(-2, 2, N), rng.uniform(0.2, 1, N)
mb, vb = rng.uniform(-2, 2, N), rng.uniform(0.2, 1, N)
mc, vc = rng.uniform(-2, 2, N), rng.uniform(0.2, 1, N)

def f32(a): return a.astype(np.float32)   # Metal (and most devices) have no double

A = BayesTensor(Tensor(f32(ma)), Tensor(f32(va)))
B = BayesTensor(Tensor(f32(mb)), Tensor(f32(vb)))
C = BayesTensor(Tensor(f32(mc)), Tensor(f32(vc)))
L = (A * B) * C                       # random loss; keep a Tensor graph on the means
L.mean.sum().backward()               # autodiff on the mean graph (scalar loss)
check("dE[L]/dmu_a == mb*mc", A.mean.grad.numpy(), (mb * mc))
check("dE[L]/dmu_b == ma*mc", B.mean.grad.numpy(), (ma * mc))
check("dE[L]/dmu_c == ma*mb", C.mean.grad.numpy(), (ma * mb))

# ---------------------------------------------------------------------------
# Section 3: Var[gradient] -- propagate moments through the gradient graph.
# grad of the RANDOM loss w.r.t. a is exactly b*c. Its moments must satisfy
#   E[grad_a]  = mb*mc
#   Var[grad_a] = mb^2*vc + mc^2*vb + vb*vc
# and must match Monte Carlo over samples of (b, c).
# ---------------------------------------------------------------------------
print("\n== 3. variance of the gradient ==")
bt_map = {B.mean.uop: B, C.mean.uop: C, A.mean.uop: A}   # forward tensor -> its moments
# grad of the random loss L=a*b*c w.r.t. each parameter, and its closed-form moments
targets = {
  "a": (A, mb * mc, mb**2 * vc + mc**2 * vb + vb * vc),  # dL/da = b*c
  "b": (B, ma * mc, ma**2 * vc + mc**2 * va + va * vc),  # dL/db = a*c
  "c": (C, ma * mb, ma**2 * vb + mb**2 * va + va * vb),  # dL/dc = a*b
}
for name, (tgt, wantE, wantV) in targets.items():
  grad_uop = compute_gradient(L.mean.uop, L.mean.uop.const_like(1.0), {tgt.mean.uop})[tgt.mean.uop]
  gm = moments(grad_uop, bt_map)                       # rebuild grad graph as BayesTensor
  if name == "a": gm_a = gm
  print(f"  grad_wrt_{tgt.mean.numpy().tolist()[:1]}... -> E={gm.mean.numpy()[:2].tolist()} Var={gm.var.numpy()[:2].tolist()}")
  check(f"E[grad_{name}] == closed form", gm.mean.numpy(), wantE)
  check(f"Var[grad_{name}] == closed form", gm.var.numpy(), wantV)

# Monte Carlo: sample the random values, differentiate the random loss by hand.
NS = 400_000
bs = rng.normal(mb, np.sqrt(vb), (NS, N))
cs = rng.normal(mc, np.sqrt(vc), (NS, N))
mc_grad_a = (bs * cs).mean(0)      # empirical E[dL/da]
mc_var_a = (bs * cs).var(0)        # empirical Var[dL/da]
check("MonteCarlo E[grad_a] == propagated", mc_grad_a, gm_a.mean.numpy(), tol=3e-3)
check("MonteCarlo Var[grad_a] == propagated", mc_var_a, gm_a.var.numpy(), tol=1e-2)  # variance estimates are noisier

# ---------------------------------------------------------------------------
# Section 4: E[gradient] through a real matmul (movement + reduce in the graph).
# loss = sum(x @ w), so dE[loss]/dw = ones @ x = column sums, broadcast per column.
# ---------------------------------------------------------------------------
print("\n== 4. expected gradient through matmul ==")
Tensor.manual_seed(1)
x = Tensor.rand(2, 3)
w = BayesTensor(Tensor.rand(3, 4), Tensor.rand(3, 4))
y = BayesTensor(x, x.const_like(0)) @ w   # deterministic input == zero-variance BayesTensor
loss = y.sum()
loss.mean.backward()
expected = (Tensor.ones(2) @ x).unsqueeze(1).expand(3, 4)
check("dE[loss]/dw == ones@x (broadcast)", w.mean.grad.numpy(), expected.numpy())

# ---------------------------------------------------------------------------
# Section 5: grad_fxn wrapping a custom fast kernel (flash-attention-style).
# A fused softmax kernel with the backward RECOMPUTED from the output (the flash
# attention trick -- no saved intermediates), then a fused (mean, var) moment
# kernel whose grad_fxn returns gradients for ALL four moments, verified for
# both E[grad] and Var[grad] through the fused backward graph.
# ---------------------------------------------------------------------------
print("\n== 5. grad_fxn wrapping a fused custom kernel ==")
from tinygrad.uop.ops import KernelInfo

def fused_softmax_kernel(Y:UOp, X:UOp, M:UOp, S:UOp) -> UOp:
  # one thread per element; row max/sum arrive as (n,1) M, S from tinygrad reduce kernels
  n, d = X.shape
  Y, X = Y.flatten(), X.flatten()
  i = UOp.range(Y.numel(), 0)
  return Y[i].store((X[i] - M[i // d, 0]).exp() / S[i // d, 0]).end(i).sink(arg=KernelInfo(name="fused_softmax"))

def fused_softmax(x:Tensor) -> Tensor:
  m = x.max(-1, keepdim=True)
  s = (x - m).exp().sum(-1, keepdim=True)
  y = Tensor.empty_like(x)
  y_out = y   # rebound to the kernel OUTPUT (an AFTER of the CALL); captured by grad_softmax below
  def grad_softmax(dy:UOp, call:UOp):
    yd = Tensor(dy)
    # flash-style: dL/dx = y * (dy - rowsum(y*dy)), recomputed from the OUTPUT only
    dldx = y_out * (yd - (y_out * yd).sum(-1, keepdim=True))
    return (None, dldx.uop, None, None)   # grads w.r.t. (Y, X, M, S)
  y_out = Tensor.custom_kernel(y, x, m, s, fxn=fused_softmax_kernel, grad_fxn=grad_softmax)[0]
  return y_out

n, d = 5, 4
Tensor.manual_seed(7)
x = Tensor.randn(n, d)
y = fused_softmax(x)
y.square().sum().backward()                      # backward BEFORE any realize()/numpy()
g_fused = x.grad.numpy()
check("fused softmax == unfused softmax", y.numpy(), x.softmax(-1).numpy())
x2 = Tensor(x.numpy())
x2.softmax(-1).square().sum().backward()
check("fused softmax backward == autodiff", g_fused, x2.grad.numpy())

# --- fused (mean, var) moment kernel: grad_fxn returns all four moment gradients ---
def fused_moments_kernel(E:UOp, V:UOp, A:UOp, VA:UOp, B:UOp, VB:UOp) -> UOp:
  E, V, A, VA, B, VB = (u.flatten() for u in (E, V, A, VA, B, VB))
  i = UOp.range(E.numel(), 0)
  return UOp.group(
    E[i].store(A[i] * B[i]),                                    # E = A*B
    V[i].store(A[i] * A[i] * VB[i] + B[i] * B[i] * VA[i] + VA[i] * VB[i]),
  ).end(i).sink(arg=KernelInfo(name="fused_moments"))

def grad_fused_moments(dE:UOp, dV:UOp, call:UOp):
  _e, _v, a, va, b, vb = call.src[1:]
  A, VA, B, VB = Tensor(a), Tensor(va), Tensor(b), Tensor(vb)
  # grads of E+V (both call outputs are used by the loss, so grad_fxn gets both dE, dV)
  gA  = Tensor(dE) * B + Tensor(dV) * (2 * A * VB)
  gVA = Tensor(dV) * (B * B + VB)
  gB  = Tensor(dE) * A + Tensor(dV) * (2 * B * VA)
  gVB = Tensor(dV) * (A * A + VA)
  return (None, None, gA.uop, gVA.uop, gB.uop, gVB.uop)   # (E, V, A, VA, B, VB)

def fused_mul(a:BayesTensor, b:BayesTensor) -> BayesTensor:
  E, V = Tensor.empty_like(a.mean), Tensor.empty_like(a.var)
  E, V, *_ = Tensor.custom_kernel(E, V, a.mean, a.var, b.mean, b.var, fxn=fused_moments_kernel, grad_fxn=grad_fused_moments)
  return BayesTensor(E, V)

A5 = BayesTensor(Tensor(f32(ma)), Tensor(f32(va)))
B5 = BayesTensor(Tensor(f32(mb)), Tensor(f32(vb)))
L5 = fused_mul(A5, B5)

# ALL symbolic work happens before any numpy()/realize() below: backward fills the
# E[grad] checks and compute_gradient gives us the RANDOM gradient graph for Var[grad].
(L5.mean.sum() + L5.var.sum()).backward()     # loss touches BOTH outputs -> grad_fxn(dE, dV, call)
L5u = L5.mean + L5.var
grad_a5 = compute_gradient(L5u.uop, L5u.uop.const_like(1.0), {A5.mean.uop})[A5.mean.uop]
bt_map5 = {A5.mean.uop: A5, B5.mean.uop: B5,
           A5.var.uop: BayesTensor(A5.var, A5.var.const_like(0)),   # variance tensors are deterministic
           B5.var.uop: BayesTensor(B5.var, B5.var.const_like(0))}
gm5 = moments(grad_a5, bt_map5)

check("dE[L]/dma == mb + 2 ma vb", A5.mean.grad.numpy(), f32(mb + 2 * ma * vb))
check("dE[L]/dva == mb^2 + vb", A5.var.grad.numpy(), f32(mb**2 + vb))
check("dE[L]/dmb == ma + 2 mb va", B5.mean.grad.numpy(), f32(ma + 2 * mb * va))
check("dE[L]/dvb == ma^2 + va", B5.var.grad.numpy(), f32(ma**2 + va))
check("fused E == ma*mb", L5.mean.numpy(), f32(ma * mb))
check("fused V == ma^2 vb + mb^2 va + va vb", L5.var.numpy(), f32(ma**2 * vb + mb**2 * va + va * vb))
check("E[grad_a] through fused bwd == mb + 2 ma vb", gm5.mean.numpy(), f32(mb + 2 * ma * vb))
check("Var[grad_a] through fused bwd == vb + 4 va vb^2", gm5.var.numpy(), f32(vb + 4 * va * vb**2))

NS = 400_000
sa = rng.normal(ma, np.sqrt(va), (NS, N))
sb = rng.normal(mb, np.sqrt(vb), (NS, N))
mc_g = (sb + 2 * sa * vb).mean(0)
mc_v = (sb + 2 * sa * vb).var(0)   # grad_a = b + 2 a vb
check("MonteCarlo E[grad_a] == propagated", mc_g, gm5.mean.numpy(), tol=3e-3)
check("MonteCarlo Var[grad_a] == propagated", mc_v, gm5.var.numpy(), tol=1e-2)

print("\nall checks passed")
