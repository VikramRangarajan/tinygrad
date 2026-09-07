import unittest

import numpy as np
from tinygrad import Tensor, UOp
from tinygrad.mixin.gradient import compute_gradient
from tinygrad.uop import Ops, GroupOp
from tinygrad.uop.ops import KernelInfo

from extra.bayesian_tensor.bayes_tensor import BayesTensor

# ---------------------------------------------------------------------------
# moments(): rebuild an arbitrary UOp expression as a BayesTensor, pairing each
# forward tensor with its variance. Used to propagate E/Var through the
# *gradient graph* -- i.e. to get E[grad] and Var[grad].
# ---------------------------------------------------------------------------
def moments(u: UOp, bt_map: dict[UOp, BayesTensor]) -> BayesTensor:
  if u in bt_map:
    return bt_map[u]
  if u.op is Ops.CONST:
    return BayesTensor(Tensor.const(u.dtype, u.arg), Tensor.const(u.dtype, 0))
  if u.op is Ops.CAST:
    m = moments(u.src[0], bt_map)
    return BayesTensor(m.expected_value.cast(u.dtype), m.variance.cast(u.dtype))
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
  if u.op in {Ops.AFTER, Ops.STORE}:
    return moments(u.src[1], bt_map)
  if u.op is Ops.COPY:
    return moments(u.src[0], bt_map)
  if u.op is Ops.CONTIGUOUS:
    return moments(u.src[0], bt_map).contiguous()
  raise NotImplementedError(f"moments: unhandled {u.op}")

def f32(a):
  return a.astype(np.float32)  # Metal (and most devices) have no double

def _abc_data():
  # shared fixture for sections 2, 3, 5: L = (a*b)*c moments
  rng = np.random.default_rng(0)
  N = 4
  ma, va = rng.uniform(-2, 2, N), rng.uniform(0.2, 1, N)
  mb, vb = rng.uniform(-2, 2, N), rng.uniform(0.2, 1, N)
  mc, vc = rng.uniform(-2, 2, N), rng.uniform(0.2, 1, N)
  return (ma, va, mb, vb, mc, vc), rng

def _abc_tensors():
  (ma, va, mb, vb, mc, vc), _ = _abc_data()
  A = BayesTensor(Tensor(f32(ma)), Tensor(f32(va)))
  B = BayesTensor(Tensor(f32(mb)), Tensor(f32(vb)))
  C = BayesTensor(Tensor(f32(mc)), Tensor(f32(vc)))
  return A, B, C

class TestForwardMoments(unittest.TestCase):
  def test_matmul_variance_closed_form(self):
    Tensor.manual_seed(42)
    a = BayesTensor(Tensor.rand(2, 3), Tensor.rand(2, 3))
    b = BayesTensor(Tensor.rand(3, 4), Tensor.rand(3, 4))
    c = a @ b
    self.assertEqual(c.expected_value.shape, (2, 4))
    self.assertEqual(c.variance.shape, (2, 4))
    # closed form: Var[A@B] = sum_k (ma_ik^2 Vb_kj + mb_kj^2 Va_ik + Va_ik Vb_kj)
    want = (a.expected_value * a.expected_value) @ b.variance + \
           a.variance @ (b.expected_value * b.expected_value) + a.variance @ b.variance
    np.testing.assert_allclose(c.variance.numpy(), want.numpy(), atol=1e-4, rtol=1e-4)

  def test_reciprocal_shapes(self):
    Tensor.manual_seed(42)
    a = BayesTensor(Tensor.rand(2, 3), Tensor.rand(2, 3))
    b = BayesTensor(Tensor.rand(3, 4), Tensor.rand(3, 4))
    r = (a @ b).reciprocal()  # nonlinear op, Taylor rule
    self.assertEqual(r.expected_value.shape, (2, 4))
    self.assertEqual(r.variance.shape, (2, 4))

  def test_stack(self):
    Tensor.manual_seed(42)
    a = BayesTensor(Tensor.rand(2, 3), Tensor.rand(2, 3))
    a2 = BayesTensor(Tensor.rand(2, 3), Tensor.rand(2, 3))
    s = a.stack(a2, dim=0)
    np.testing.assert_allclose(s.expected_value.numpy(), np.stack([a.expected_value.numpy(), a2.expected_value.numpy()]),
                               atol=1e-6, rtol=1e-6)
    self.assertTrue((s.variance.numpy() >= 0).all())

  def test_detach_noop(self):
    Tensor.manual_seed(42)
    c = BayesTensor(Tensor.rand(2, 3), Tensor.rand(2, 3)) @ BayesTensor(Tensor.rand(3, 4), Tensor.rand(3, 4))
    d = c.detach()
    np.testing.assert_allclose(d.expected_value.numpy(), c.expected_value.numpy(), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(d.variance.numpy(), c.variance.numpy(), atol=1e-6, rtol=1e-6)

class TestExpectedGradient(unittest.TestCase):
  def test_expected_gradient_product(self):
    # L = (a*b)*c, so dE[L]/dmu_a = mu_b * mu_c
    (ma, _, mb, _, mc, _), _ = _abc_data()
    A, B, C = _abc_tensors()
    L = (A * B) * C  # random loss; keep a Tensor graph on the means
    L.expected_value.sum().backward()  # autodiff on the mean graph (scalar loss)
    np.testing.assert_allclose(A.expected_value.grad.numpy(), mb * mc, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(B.expected_value.grad.numpy(), ma * mc, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(C.expected_value.grad.numpy(), ma * mb, atol=1e-4, rtol=1e-4)

class TestGradientVariance(unittest.TestCase):
  def _grad_moments(self, name):
    A, B, C = _abc_tensors()
    (ma, va, mb, vb, mc, vc), _ = _abc_data()
    L = (A * B) * C
    bt_map = {B._uop: B, C._uop: C, A._uop: A}  # forward tensor -> its moments
    targets = {
      # grad of the random loss L=a*b*c w.r.t. each parameter, and its closed-form moments
      "a": (A, mb * mc, mb**2 * vc + mc**2 * vb + vb * vc),  # dL/da = b*c
      "b": (B, ma * mc, ma**2 * vc + mc**2 * va + va * vc),  # dL/db = a*c
      "c": (C, ma * mb, ma**2 * vb + mb**2 * va + va * vb),  # dL/dc = a*b
    }
    tgt, wantE, wantV = targets[name]
    grad_uop = compute_gradient(L._uop, L._uop.const_like(1.0), {tgt._uop})[tgt._uop]
    gm = moments(grad_uop, bt_map)  # rebuild grad graph as BayesTensor
    return gm, wantE, wantV

  def test_moments_match_closed_form(self):
    for name in ("a", "b", "c"):
      with self.subTest(grad=name):
        gm, wantE, wantV = self._grad_moments(name)
        np.testing.assert_allclose(gm.expected_value.numpy(), wantE, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(gm.variance.numpy(), wantV, atol=1e-4, rtol=1e-4)

  def test_monte_carlo_grad_a(self):
    gm, _, _ = self._grad_moments("a")
    (ma, va, mb, vb, mc, vc), rng = _abc_data()
    NS = 400_000
    bs = rng.normal(mb, np.sqrt(vb), (NS, len(mb)))
    cs = rng.normal(mc, np.sqrt(vc), (NS, len(mc)))
    np.testing.assert_allclose((bs * cs).mean(0), gm.expected_value.numpy(), atol=3e-3, rtol=3e-3)
    # variance estimates are noisier
    np.testing.assert_allclose((bs * cs).var(0), gm.variance.numpy(), atol=1e-2, rtol=1e-2)

class TestMatmulExpectedGradient(unittest.TestCase):
  def test_expected_gradient_through_matmul(self):
    # loss = sum(x @ w), so dE[loss]/dw = ones @ x = column sums, broadcast per column
    Tensor.manual_seed(1)
    x = Tensor.rand(2, 3)
    w = BayesTensor(Tensor.rand(3, 4), Tensor.rand(3, 4))
    y = BayesTensor(x, x.const_like(0)) @ w  # deterministic input == zero-variance BayesTensor
    y.sum().expected_value.backward()
    expected = (Tensor.ones(2) @ x).unsqueeze(1).expand(3, 4)
    np.testing.assert_allclose(w.expected_value.grad.numpy(), expected.numpy(), atol=1e-4, rtol=1e-4)

def fused_softmax_kernel(Y: UOp, X: UOp, M: UOp, S: UOp) -> UOp:
  # one thread per element; row max/sum arrive as (n,1) M, S from tinygrad reduce kernels
  n, d = X.shape
  Y, X = Y.flatten(), X.flatten()
  i = UOp.range(Y.numel(), 0)
  return Y[i].store((X[i] - M[i // d, 0]).exp() / S[i // d, 0]).end(i).sink(arg=KernelInfo(name="fused_softmax"))

def fused_softmax(x: Tensor) -> Tensor:
  m = x.max(-1, keepdim=True)
  s = (x - m).exp().sum(-1, keepdim=True)
  y = Tensor.empty_like(x)
  y_out = y  # rebound to the kernel OUTPUT (an AFTER of the CALL); captured by grad_softmax below

  def grad_softmax(dy: UOp, call: UOp):
    yd = Tensor(dy)
    # flash-style: dL/dx = y * (dy - rowsum(y*dy)), recomputed from the OUTPUT only
    dldx = y_out * (yd - (y_out * yd).sum(-1, keepdim=True))
    return (None, dldx.uop, None, None)  # grads w.r.t. (Y, X, M, S)

  y_out = Tensor.custom_kernel(y, x, m, s, fxn=fused_softmax_kernel, grad_fxn=grad_softmax)[0]
  return y_out

def fused_moments_kernel(E: UOp, V: UOp, A: UOp, VA: UOp, B: UOp, VB: UOp) -> UOp:
  E, V, A, VA, B, VB = (u.flatten() for u in (E, V, A, VA, B, VB))
  i = UOp.range(E.numel(), 0)
  return (
    UOp
    .group(
      E[i].store(A[i] * B[i]),  # E = A*B
      V[i].store(A[i] * A[i] * VB[i] + B[i] * B[i] * VA[i] + VA[i] * VB[i]),
    )
    .end(i)
    .sink(arg=KernelInfo(name="fused_moments"))
  )

def grad_fused_moments(dE: UOp, dV: UOp, call: UOp):
  _e, _v, a, va, b, vb = call.src[1:]
  A, VA, B, VB = Tensor(a), Tensor(va), Tensor(b), Tensor(vb)
  # grads of E+V (both call outputs are used by the loss, so grad_fxn gets both dE, dV)
  gA = Tensor(dE) * B + Tensor(dV) * (2 * A * VB)
  gVA = Tensor(dV) * (B * B + VB)
  gB = Tensor(dE) * A + Tensor(dV) * (2 * B * VA)
  gVB = Tensor(dV) * (A * A + VA)
  return (None, None, gA.uop, gVA.uop, gB.uop, gVB.uop)  # (E, V, A, VA, B, VB)

def fused_mul(a: BayesTensor, b: BayesTensor) -> BayesTensor:
  E, V = Tensor.empty_like(a.expected_value), Tensor.empty_like(a.variance)
  E, V, *_ = Tensor.custom_kernel(
    E, V, a.expected_value, a.variance, b.expected_value, b.variance, fxn=fused_moments_kernel, grad_fxn=grad_fused_moments
  )
  return BayesTensor(E, V)

class TestFusedKernels(unittest.TestCase):
  def test_fused_softmax_forward_backward(self):
    n, d = 5, 4
    Tensor.manual_seed(7)
    x = Tensor.randn(n, d)
    y = fused_softmax(x)
    y.square().sum().backward()  # backward BEFORE any realize()/numpy()
    g_fused = x.grad.numpy()
    np.testing.assert_allclose(y.numpy(), x.softmax(-1).numpy(), atol=1e-4, rtol=1e-4)
    x2 = Tensor(x.numpy())
    x2.softmax(-1).square().sum().backward()
    np.testing.assert_allclose(g_fused, x2.grad.numpy(), atol=1e-4, rtol=1e-4)

  def test_fused_moments_forward_and_expected_grad(self):
    (ma, va, mb, vb, _, _), _ = _abc_data()
    A5 = BayesTensor(Tensor(f32(ma)), Tensor(f32(va)))
    B5 = BayesTensor(Tensor(f32(mb)), Tensor(f32(vb)))
    L5 = fused_mul(A5, B5)
    # ALL symbolic work happens before any numpy()/realize() below: backward fills the
    # E[grad] checks and compute_gradient gives us the RANDOM gradient graph for Var[grad].
    (L5.expected_value.sum() + L5.variance.sum()).backward()  # loss touches BOTH outputs -> grad_fxn(dE, dV, call)
    np.testing.assert_allclose(A5.expected_value.grad.numpy(), f32(mb + 2 * ma * vb), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(A5.variance.grad.numpy(), f32(mb**2 + vb), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(B5.expected_value.grad.numpy(), f32(ma + 2 * mb * va), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(B5.variance.grad.numpy(), f32(ma**2 + va), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(L5.expected_value.numpy(), f32(ma * mb), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(L5.variance.numpy(), f32(ma**2 * vb + mb**2 * va + va * vb), atol=1e-4, rtol=1e-4)

  def test_fused_moments_gradient_variance(self):
    (ma, va, mb, vb, _, _), _ = _abc_data()
    A5 = BayesTensor(Tensor(f32(ma)), Tensor(f32(va)))
    B5 = BayesTensor(Tensor(f32(mb)), Tensor(f32(vb)))
    L5 = fused_mul(A5, B5)
    (L5.expected_value.sum() + L5.variance.sum()).backward()
    L5u = L5.expected_value + L5.variance
    grad_a5 = compute_gradient(L5u.uop, L5u.uop.const_like(1.0), {A5._uop})[A5._uop]
    bt_map5 = {
      A5._uop: A5,
      B5._uop: B5,
      A5.variance.uop: BayesTensor(A5.variance, A5.variance.const_like(0)),  # variance tensors are deterministic
      B5.variance.uop: BayesTensor(B5.variance, B5.variance.const_like(0)),
    }
    gm5 = moments(grad_a5, bt_map5)
    np.testing.assert_allclose(gm5.expected_value.numpy(), f32(mb + 2 * ma * vb), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(gm5.variance.numpy(), f32(vb + 4 * va * vb**2), atol=1e-4, rtol=1e-4)

  def test_fused_moments_monte_carlo(self):
    (ma, va, mb, vb, _, _), rng = _abc_data()
    NS = 400_000
    sa = rng.normal(ma, np.sqrt(va), (NS, len(ma)))
    sb = rng.normal(mb, np.sqrt(vb), (NS, len(mb)))
    # grad_a = b + 2 a vb
    A5 = BayesTensor(Tensor(f32(ma)), Tensor(f32(va)))
    B5 = BayesTensor(Tensor(f32(mb)), Tensor(f32(vb)))
    L5 = fused_mul(A5, B5)
    L5u = L5.expected_value + L5.variance
    grad_a5 = compute_gradient(L5u.uop, L5u.uop.const_like(1.0), {A5._uop})[A5._uop]
    bt_map5 = {
      A5._uop: A5,
      B5._uop: B5,
      A5.variance.uop: BayesTensor(A5.variance, A5.variance.const_like(0)),
      B5.variance.uop: BayesTensor(B5.variance, B5.variance.const_like(0)),
    }
    gm5 = moments(grad_a5, bt_map5)
    np.testing.assert_allclose((sb + 2 * sa * vb).mean(0), gm5.expected_value.numpy(), atol=3e-3, rtol=3e-3)
    np.testing.assert_allclose((sb + 2 * sa * vb).var(0), gm5.variance.numpy(), atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
  unittest.main()
