from tinygrad import Tensor
from tinygrad.callify import transform_to_call  # noqa
from tinygrad.engine.realize import run_linear, link_linear, compile_linear  # noqa
from tinygrad.mixin.elementwise import ElementwiseMixin  # noqa
from tinygrad.mixin.rand import RandMixin  # noqa
from tinygrad.mixin.reduce import ReduceMixin  # noqa
from tinygrad.schedule import create_linear_with_vars  # noqa
from tinygrad.uop import Ops  # noqa
from tinygrad.uop.ops import graph_rewrite  # noqa
from tinygrad.codegen import do_to_program  # noqa


def _path():
    other = Tensor([])
    Tensor.__add__
    Tensor.add
    Tensor._binop(Ops.ADD, other, reverse=False)
    Tensor._broadcasted(other, reverse=False) # gives broadcasted x and y
    Tensor.alu(other)
    op = Ops.ADD
    Tensor._apply_uop(lambda *u: u[0].alu(op, *u[1:]), other) # creates tensor with uop add

    Tensor.realize
    Tensor.linear_with_vars  # Creates new UOp.sink with every single realize tensor
    transform_to_call
    graph_rewrite # ? Just accept it
# Investigate AllocCtx, UOp.is_virtual, UOp.has_buffer_identity
    create_linear_with_vars

    run_linear
    compile_linear
    # If beam, then do beam pattern matching
    # For pm_compile, pass Call(Sink) UOp into to_program and replace
    link_linear  # noop without hcq2


a = Tensor.randn(10**6)
b = Tensor.randn(10**6)
c = a + b
c.realize()
