# New session - 2026-07-30T04:52:52.253Z

**Session ID:** ses_04ea1be62ffer0g73k0dUMqtw5
**Created:** 7/30/2026, 10:22:52 AM
**Updated:** 7/30/2026, 10:50:36 AM

---

## User

I want to understand how tinygrad works. The whole codebase from top down. Please give me a study guide on how to inspect and play with the codebase. I already have a full understanding of how ndarrays / autodiff works. I have a small understanding of how graph JIT works on a computation DAG. I'm just not sure how tinygrad does all of this. Keep this in mind.

---

## Assistant

**Tool: read**

**Tool: task**

---

## Assistant

Here's your study guide, structured as a top-down reading path with hands-on exercises.

---

## tinygrad Study Guide

### Philosophy

tinygrad is built around **one universal IR node** (`UOp`). Everything — tensors, operations, kernels, memory, gradients — is a `UOp`. There is **no separate Autograd graph, no separate IR, no separate scheduler graph**. One type to rule them all.

---

### Phase 1: Core Abstractions (read in order)

| Step | File | What to look for |
|---|---|---|
| 1 | `tinygrad/uop/ops.py:142` | `class UOp` — the one node. Fields: `op`, `dtype`, `src` (tuple of inputs), `arg`. Immutable via metaclass + interning. |
| 2 | `tinygrad/uop/__init__.py:13` | `class Ops(FastEnum)` — all ops in the system. `Ops.ADD`, `Ops.STORE`, `Ops.REDUCE`, `Ops.RESHAPE`, etc. |
| 3 | `tinygrad/dtype.py:57` | `class DType` — type system. Scalar + `PtrDType` (pointer wrapper) + `ImageDType`. |
| 4 | `tinygrad/device.py:15` | `Device` singleton: `Device["METAL"]` creates a `Compiled` object. Lazy-loads `tinygrad.runtime.ops_*`. |
| 5 | `tinygrad/device.py:99` | `class Buffer` — raw device memory. `allocate()`, `copyin()`, `copyout()`, `view()`. |
| 6 | `tinygrad/tensor.py:62` | `class Tensor` — user-facing NDArray. Wraps `self.uop: UOp`. |

**Mental model:** `Tensor` is the *public API*. Underneath, it holds a `UOp` DAG. Operations on `Tensor` build new `UOp` nodes. No eager execution — just graph construction.

**Exercise:**
```python
from tinygrad import Tensor, dtypes
a = Tensor([1,2,3])
b = Tensor([4,5,6])
c = a + b
print(c.uop)           # <UOp Ops.ADD>
print(c.uop.src)       # (UOp for a, UOp for b)
# Look at the DAG:
print(c.uop.toposort)
```

---

### Phase 2: How Ops Work (follow the data)

When you write `a + b`, here's the chain:

1. `Tensor.__add__` → calls `OpMixin._arithmetic_op()` → `tensor.py`
2. That calls `self._apply_uop(ops.ADD, ...)` → creates `UOp(Ops.ADD, src=(a.uop, b.uop))`
3. The new `Tensor` wraps this `UOp` — **no computation yet**.

The `mixin/` directory has all the op implementations:
- `mixin/elementwise.py` — `_arithmetic_op`, `_comparison_op`, `_unary_op`
- `mixin/movement.py` — reshape, permute, expand, pad, shrink, flip
- `mixin/reduce.py` — `_reduce_op` (sum, max, etc.)

**Exercise:** Trace through adding two tensors with a debugger or print statements. Watch `UOp` nodes accumulate.

---

### Phase 3: Realization / Execution

Nothing runs until you call `.numpy()`, `.item()`, or `Tensor.realize()`.

1. **Scheduling** — `schedule/__init__.py:create_linear_with_vars()`
   - Takes the final `UOp` DAG (a `SINK` node)
   - Decomposes it into ordered kernels (each kernel is a `LINEAR` UOp with `STORE`/`LOAD`/`REDUCE`)
   - Returns a topological schedule

2. **Realization** — `engine/realize.py`
   - Each kernel in the schedule gets `lower()`ed → `codegen/__init__.py:to_program()`
   - Codegen pipeline (simplify → expand → devectorize → assign dims → render → compile)
   - Result: `ProgramInfo` (kernel binary + metadata)
   - Kernels are dispatched via `Compiled.runtime()` or `Compiled.graph()` for graph batching

3. **Execution** — the rendered kernel runs on the device, reads/writes `Buffer`s.

**Exercise:**
```python
from tinygrad import Tensor
a = Tensor([1,2,3]); b = Tensor([4,5,6])
c = a + b
# Before realize: c.uop is a DAG node, no computation
c = c.realize()        # forces scheduling + compilation + execution
print(c.numpy())       # realized result
```

---

### Phase 4: JIT Compilation (`TinyJit`)

File: `engine/jit.py`

How `@TinyJit` works:

1. **First call:** Captures the schedule (list of kernels produced by `create_linear_with_vars`). Records input buffer params.
2. **jit_lower()**: Substitutes real buffers with `PARAM` UOps. Compiles each kernel to `ProgramInfo`. Runs `memory_plan_rewrite()` for sub-allocation. Optionally batches consecutive kernels into `CUSTOM_FUNCTION` GPU graphs.
3. **Subsequent calls:** Replays the compiled `ProgramInfo` list with new input buffers. No scheduling, no compilation — just dispatch.

**Exercise:**
```python
from tinygrad import Tensor, TinyJit

@TinyJit
def f(x, y):
    return (x + y).realize()

for _ in range(5):
    a = Tensor([1,2,3]); b = Tensor([4,5,6])
    out = f(a, b)       # first call compiles, rest replay
    print(out.numpy())
```

Set `DEBUG=1` or `DEBUG=2` to see what's happening:
```bash
DEBUG=2 python your_script.py
```

---

### Phase 5: Gradients

File: `gradient.py`

`compute_gradient(root_uop, root_grad, targets)` — reverse-mode AD on `UOp` graphs.

- Uses `PatternMatcher` to define gradient rules per-op
- Each `Ops` value has a rule: given output gradient, compute input gradients
- `Tensor.backward()` calls `compute_gradient(self.uop, const_like(1.0), ...)`
- Works on the **unrealized UOp graph** — builds a gradient *graph*, then later realizes it

**Exercise:**
```python
from tinygrad import Tensor
x = Tensor([1., 2., 3.], requires_grad=True)
y = (x * 2).sum()
y.backward()
print(x.grad)           # [2., 2., 2.]
# Inspect the gradient graph:
print(x.grad.uop)
```

---

### Phase 6: Codegen Pipeline (deep dive)

File: `codegen/__init__.py:to_program()`

Traces the full pipeline from UOp DAG → source code:

1. **Early opt:** fold constants, syntactic sugar
2. **Load collapse:** optimize tensor indexing (merge adjacent loads)
3. **Range splitting:** split loops into GLOBAL/LOCAL/THREAD ranges
4. **Symbolic simplification:** `uop/symbolic.py` — rewrite `(x+1)*2` → `x*2+2`
5. **Post-range opt:** BEAM search for best kernel config
6. **Expander** (`codegen/late/expander.py`): unroll constant loops, expand shapes
7. **Devectorizer** (`codegen/late/devectorizer.py`): remove vector ops, add scalar loads/stores
8. **GPU dims** (`codegen/gpudims.py`): assign workgroup dimensions
9. **Render** (`renderer/cstyle.py`): emit C-like code (Metal, CUDA, OpenCL, etc.)
10. **Compile** (`device.py:Compiler`): compile source → binary (via clang, metal-lang, nvcc, etc.)

**Exercise:** Print the rendered kernel source:
```python
import os; os.environ["RENDER_HUMAN"] = "1"
from tinygrad import Tensor
a = Tensor([1,2,3]); b = Tensor([4,5,6])
c = (a + b).realize()
# With DEBUG=2, the kernel source prints to stdout
```

---

### Phase 7: Visualization & Debugging

| Tool | How |
|---|---|
| `DEBUG=1` | Print schedule (which kernels run, shapes) |
| `DEBUG=2` | Print kernel source code |
| `DEBUG=4` | Print symbolic expressions |
| `DEBUG=5` | Print gradient graphs |
| `RENDER_HUMAN=1` | Human-readable kernel output |
| `SAVE_KERNELS=1` | Dump GPU binaries to disk |

---

### Suggested Reading Order (files to study)

```
1.  tinygrad/uop/__init__.py       (understand the Ops vocabulary)
2.  tinygrad/uop/ops.py            (UOp class — the universal node)
3.  tinygrad/dtype.py              (type system)
4.  tinygrad/device.py             (Device, Buffer, Compiled, Allocator)
5.  tinygrad/tensor.py             (Tensor class — user API)
6.  tinygrad/mixin/elementwise.py  (how ops get attached to Tensor)
7.  tinygrad/mixin/movement.py     (reshape/permute/etc as UOp graphs)
8.  tinygrad/mixin/reduce.py       (reductions as UOp graphs)
9.  tinygrad/schedule/__init__.py  (kernel scheduling: create_linear_with_vars)
10. tinygrad/codegen/__init__.py   (full compilation pipeline)
11. tinygrad/engine/realize.py     (kernel realization)
12. tinygrad/engine/jit.py         (TinyJit — graph capture + replay)
13. tinygrad/gradient.py           (autodiff on UOp graphs)
14. tinygrad/function.py           (@function decorator — custom fused kernels)
15. tinygrad/callify.py            (graph → callable kernel lowering)
```

---

### Key Insight: The `UOp` Graph is *Everything*

Compare to PyTorch:
- **PyTorch:** Eager execution (per-op kernel launch). Autograd graph is separate metadata on tensors. JIT has its own IR.
- **tinygrad:** Build a `UOp` DAG lazily. Gradients are more `UOp`s. Scheduling decomposes `UOp` DAG into kernels. Codegen renders `UOp`s to source. **One IR throughout.**

The pipeline is:

```
Tensor ops → UOp DAG (lazy, no compute)
                ↓
        + gradients (more UOp DAG nodes)
                ↓
        schedule (split into kernel-sized UOp subgraphs)
                ↓
        lower/codegen (UOp → source → binary)
                ↓
        dispatch (GPU launch)
```

When you understand `UOp`, you understand tinygrad. **Start by reading `uop/ops.py`** and playing with `Tensor(...).uop` in the REPL.

---

## User

Question. For gradients, is tinygrad well designed as in not to have a separate "aotautograd" system like pytorch, where the gradient graph is just appended to the main dag? And if so, can I write extension to tinygrad that can do things like calculate hessians easily? Or is autodiff very closely intertwined

---

## Assistant

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: grep**

**Tool: grep**

**Tool: grep**

---

## Assistant

**Tool: read**

**Tool: read**

---

## Assistant

Short answer to both: **yes, the design is clean — gradients are just more UOps appended to the same DAG, and Hessians are trivial.**

Here's why:

### How autodiff works in tinygrad

`compute_gradient(root, root_grad, targets)` at `gradient.py:99` is a **pure function** on UOp graphs. It:

1. Walks the graph in reverse topological order
2. For each UOp, looks up its gradient rule via `PatternMatcher` (a simple pattern-match dispatch table at `gradient.py:53-90`)
3. Each rule returns gradient *UOps* for the inputs (e.g., `Ops.MUL` → `λ ctx, ret: (ret.src[1]*ctx, ret.src[0]*ctx)`)
4. Gradients get accumulated: `grads[input] = grads[input] + new_grad`

The output is `dict[UOp, UOp]` — a mapping from forward UOps to their gradient UOps. These gradient UOps are just new nodes in the same UOp DAG. No separate "autograd engine" or "AOT trace". The dict is returned to the caller; `Tensor.backward()` assigns them to `.grad` fields.

### Hessians in ~5 lines

Since `compute_gradient` works on arbitrary UOp targets, you compute a Hessian by running autodiff on the gradient:

```python
from tinygrad import Tensor
from tinygrad.gradient import compute_gradient

x = Tensor.eye(3).requires_grad_()
y = x.sum()  # scalar
# First-order grads
grads = compute_gradient(y.uop, y.const_like(1.0).uop, {x.uop})
grad_uop = grads[x.uop]  # gradient of y w.r.t x
# Hessian: run autodiff on each output of the gradient
hessian = compute_gradient(grad_uop, grad_uop.const_like(1.0).uop, {x.uop})
```

The only entanglement is that `compute_gradient` is in `gradient.py` and calls into `function.py` for custom gradient functions (the `@function` decorator), but the API surface you need (`compute_gradient`) is fully separable. You could write:

```python
def hessian(f, x):
    y = f(x)
    grads = compute_gradient(y.uop, y.const_like(1.0).uop, {x.uop})
    g = grads[x.uop]
    # second-order: flatten g, compute jacobian of each element
    g_flat = g.flatten()
    H_rows = []
    for i in range(g_flat.numel()):
        gi = g_flat[i]
        h = compute_gradient(gi.uop, gi.const_like(1.0).uop, {x.uop})
        H_rows.append(h[x.uop].reshape(1, -1))
    return Tensor.cat(*H_rows, dim=0)
```

**Why it works**: The gradient graph is just more `UOp` nodes in the same pool. `compute_gradient` doesn't care whether the input graph is a "forward" graph or a "gradient" graph — it just applies chain rule to whatever UOps it finds. So you can nest it arbitrarily.

---

## User

Another question. How does llm inference compilation work? Since the kv cache changes shape every token generation would that need a recompilation? I have used tinygrad/llm and see the warmup option, please explain that

---

## Assistant

**Tool: read**

**Tool: glob**

---

## Assistant

**Tool: read**

---

## Assistant

**Tool: read**

**Tool: grep**

---

## Assistant

**Tool: grep**

---

## Assistant

**Tool: grep**

---

## Assistant

**Tool: grep**

**Tool: grep**

---

## Assistant

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: grep**

---

## Assistant

**Tool: read**

---

## Assistant

The key mechanism is **symbolic shapes via `UOp.variable()`**. Here's the full picture:

### How dynamic shapes avoid recompilation

In `Transformer.generate()` (line 398-399):

```python
v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
v_toks = UOp.variable("toks", 1, chunk_size)
```

These create `DEFINE_VAR` UOps — symbolic integers with known min/max ranges. When they flow into shape arithmetic (slicing the cache at `start_pos:start_pos+T`, computing attention mask size, etc.), the entire UOp graph becomes **symbolic** — shapes like `(1, 1, T, start_pos+T)` are symbolic expressions, not concrete ints.

When TinyJit captures the schedule on its first `cnt=1` call, it:

1. Captures the full UOp DAG containing these symbolic expressions
2. Compiles kernels **once** with `start_pos` and `T` as kernel parameters (passed at runtime)
3. The compiled kernel code has loop bounds like `for i < start_pos + T` where those are register values, not constants

On subsequent calls (`cnt >= 2`), only `var_vals` (the binding dict `{"start_pos": 42, "toks": 1}`) changes. The `GraphRunner.updated_vars()` and `updated_launch_dims()` methods at `jit.py:161-169` resolve the symbolic launch dimensions from the current variable values **without recompiling**.

**The cache is pre-allocated to `max_context`** (fixed buffer shape). Only the views into it change symbolically.

### The `resolve()` trick

At `model.py:180`:
```python
mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), ...).triu(start_pos+1) \
  if resolve(T != 1) else None
```

`resolve()` at `ops.py:50` evaluates whether `T != 1` is **statically knowable** — if `T` is bound to 1 (single-token decode), it returns `False` and elides the mask entirely. If `T > 1` (prefill), it returns `True`. But since T is always bound to a specific value before execution, this always resolves concretely — no branch in the compiled code.

### What "warmup" does

It's not separate compilation — it's just running a few tokens so that TinyJit goes through its **capture phase** (`cnt=0→1`). After that (`cnt>=2`), every call is pure replay:

```
cnt=0:   bypass JIT, run eagerly (no capture)
cnt=1:   run + capture the symbolic graph + compile → CapturedJit
cnt=2+:  CapturedJit.__call__ → just dispatch with new var_vals
```

The `@TinyJit` actually has **two separate instances** (`prefill_jit` and `rollout_jit` at line 310-311), selected by `resolve(tokens.shape[1] != 1)` — so prefill and decode get independently captured schedules, since they have different symbolic shapes (`T=chunk_size` vs `T=1`).

---

