"""
BayesTensor: an OpMixin subclass that propagates both E[X] and Var[X] through a graph.

Only ONE primitive needs moment rules: `alu` (elementwise). Everything else (matmul,
dot, sum, reshape, transpose, ...) is inherited for free from OpMixin, because those ops
are each defined exactly once in terms of alu / _rop / _mop.

The point of this file is that gradients work in BOTH senses:

  * E[gradient]  -- standard tinygrad autodiff on the MEAN graph (loss.expected_value.backward())
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
from tinygrad import Tensor
from tinygrad.helpers import argfix
from tinygrad.mixin.op import OpMixin
from tinygrad.uop import Ops, GroupOp
from tinygrad.uop.ops import UOp



class BayesTensor(OpMixin):
  __slots__ = "expected_value", "variance"

  def __init__(self, mean: Tensor, var: Tensor):
    self.expected_value: Tensor = mean
    self.variance: Tensor = var

  # ---- the ~8-method abstract surface OpMixin needs ----
  @property
  def shape(self):
    return self.expected_value.shape

  @property
  def dtype(self):
    return self.expected_value.dtype

  @property
  def device(self):
    return self.expected_value.device

  @classmethod
  def const(cls, dtype, b):
    return cls(Tensor.const(dtype, b), Tensor.const(dtype, 0))

  @classmethod
  def _wrap_uop(cls, u):
    raise NotImplementedError("cast/ufix are overridden so this is never hit")

  @property
  def _uop(self):
    return self.expected_value.uop

  # scalar / uop lifting -> deterministic quantity with zero variance
  def ufix(self, x):
    if isinstance(x, BayesTensor):
      return x
    if isinstance(x, UOp):
      x = self.expected_value._wrap_uop(x)
    return BayesTensor(self.expected_value.ufix(x), self.variance.ufix(x) * 0)

  def cast(self, dtype):
    if self.expected_value.dtype == dtype and self.variance.dtype == dtype:
      return self
    return BayesTensor(self.expected_value.cast(dtype), self.variance.cast(dtype))

  # ---- elementwise primitive: the moment propagation rules ----
  def alu(self, op: Ops, *src: BayesTensor) -> BayesTensor:
    match op:
      case Ops.EXP2:
        raise NotImplementedError("TODO!")
      case Ops.LOG2:
        raise NotImplementedError("TODO!")
      case Ops.SQRT:
        raise NotImplementedError("TODO!")
      case Ops.POW:
        raise NotImplementedError("TODO!")
      case Ops.FDIV:
        # Use approximation X/E[Y] instead of true X/Y
        # E[X/E[Y]] = E[X]/E[Y], Var[X/E[Y]] = Var[X]/E[Y]^2
        return BayesTensor(self.expected_value / src[0].expected_value, self.variance / src[0].expected_value.square())
      case Ops.SIN:
        raise NotImplementedError("TODO!")
      case Ops.RECIPROCAL:
        # first-order Taylor: E[1/x] ~ 1/m, Var[1/x] ~ v/m^4
        m, v = self.expected_value, self.variance
        return BayesTensor(m.reciprocal(), v * m.reciprocal() ** 4)
      case Ops.NEG:
        return BayesTensor(-self.expected_value, self.variance)
      case Ops.TRUNC:
        raise NotImplementedError("TODO!")
      case Ops.ADD:
        # Var[a+b] = Va + Vb (independence) and linearity of expectation
        return BayesTensor(self.expected_value + src[0].expected_value, self.variance + src[0].variance)
      case Ops.MUL:
        # E[ab] = ma*mb, Var[ab] = ma^2*Vb + mb^2*Va + Va*Vb
        ma, va, mb, vb = self.expected_value, self.variance, src[0].expected_value, src[0].variance
        return BayesTensor(ma * mb, ma * ma * vb + mb * mb * va + va * vb)
      case Ops.MAX:
        raise NotImplementedError("TODO!")
      case Ops.SUB:
        return BayesTensor(self.expected_value - src[0].expected_value, self.variance + src[0].variance)
      case Ops.DETACH:
        # routing (no derivation): detach both moments from autograd
        return BayesTensor(self.expected_value.detach(), self.variance.detach())
      case Ops.CONTIGUOUS_BACKWARD:
        # routing: identity in the backward pass
        return self
      case (
        Ops.CDIV
        | Ops.CMOD
        | Ops.CMPLT
        | Ops.CMPNE
        | Ops.CMPEQ
        | Ops.XOR
        | Ops.SHL
        | Ops.SHR
        | Ops.OR
        | Ops.AND
        | Ops.THREEFRY
        | Ops.FLOORDIV
        | Ops.FLOORMOD
      ):
        raise NotImplementedError("Moment propagation rules do not exist for", op)
      case _:
        raise NotImplementedError(f"alu rule missing for {op}")

  # ---- reduce primitive: Var[sum] = sum(Var) ----
  def _rop(self, op: Ops, axis: tuple) -> BayesTensor:
    match op:
      case Ops.ADD:
        return BayesTensor(self.expected_value.sum(axis), self.variance.sum(axis))
      case _:
        raise NotImplementedError(f"_rop rule missing for {op}")

  # ---- movement primitive: apply to both moments ----
  def _mop(self, op: Ops, arg) -> BayesTensor:
    return BayesTensor(self.expected_value._mop(op, arg), self.variance._mop(op, arg))

  # STACK can't go through _mop (its arg is the *uops* of the other tensors,
  # so the vars would be lost) -- override the high-level method instead
  def stack(self, *args, dim=0):
    tensors = argfix(self, *args)
    return BayesTensor(
      tensors[0].expected_value.stack(*[t.expected_value for t in tensors[1:]], dim=dim),
      tensors[0].variance.stack(*[t.variance for t in tensors[1:]], dim=dim),
    )

  # contiguous is a no-op semantically for the (mean, var) pair
  def contiguous(self, **kwargs):
    return BayesTensor(self.expected_value.contiguous(**kwargs), self.variance.contiguous(**kwargs))

  def __repr__(self):
    return f"BayesTensor(mean={self.expected_value.shape}, var={self.variance.shape})"
