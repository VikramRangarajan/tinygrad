# Public API

ElementwiseMixin, ReduceMixin
OpMixin
RandMixin
Tensor
Tensor's a + b:

- Tensor.\_\_add\_\_ -> ElementwiseMixin.add -> OpMixin.\_binop(op.ADD) -> a.alu(op.ADD, a.ufix(x))
- a.ufix(b) -> tensor.uop.ufix -> ops.const_like -> UOp.const
- Tensor.alu -> Tensor.\_apply_uop(lambda *u: u[0].alu(op, *u[1:]), \*src)
