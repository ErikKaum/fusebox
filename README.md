# Fusebox

Trace-based tensor compiler for Rust. Build computation graphs with a familiar tensor API, lower them to [StableHLO](https://github.com/openxla/stablehlo) MLIR, and execute through [PJRT](https://github.com/openxla/xla/tree/main/xla/pjrt) on CPU.

## Background

Fusebox started as an exercise in understanding how [ZML](https://github.com/zml/zml) works under the hood, the idea was to rebuild the core trace-compile-run loop from scratch in Rust and see what it actually takes to go from tensor ops to running hardware. What began as a learning project turned into something genuinely fun to hack on, and it kept growing from there.

I wrote [a blog](https://www.erikkaum.com/blog/zml/) that hopefully clarifies how the stack works and helps you build your own toy/educational ML framework

And despite being educational, it even runs the `SmolLM2-135M-Instruct` model on CPU:

```
./target/release/examples/smollm2 chat --compiled examples/smollm2/artifacts/smollm2.compiled
Loaded compiled model in 97.97ms
Loaded weights in 91.29ms

SmolLM2-135M-Instruct ready. Type a message and press Enter. Type "exit" to quit.

You> Where is L'Arc de Triomphe located?
Assistant> The Arc de Triomphe is located in the heart of Paris, France. It is a monumental arch that spans the entire length of the Eiffel Tower, connecting the top of the tower to the ground below. The Arc de Triomphe is a symbol of Paris and a popular tourist attraction.
[32 prompt tokens, 60 generated | TTFT 292ms | 4.8 tok/s]
```

## How it works

Fusebox follows a **trace → compile → run** workflow:

1. **Trace** — Write your model using symbolic `Tensor` operations. Instead of computing eagerly, each op records an instruction in a computation graph.
2. **Compile** — The graph is lowered to StableHLO MLIR and compiled into a PJRT executable for your target hardware.
3. **Run** — Feed concrete weight and input data into the compiled executable and get results back.

```rust
use fusebox::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ckpt = Checkpoint::from_file("model.safetensors")?;
    let device = Device::cpu();

    // 1. Trace & compile
    let runner = device.compile("main", |cx| {
        let x = cx.input("x", Shape::new(vec![2, 4], DType::F32));
        let linear = Linear::trace(cx, "linear", ckpt.shapes())?;
        linear.forward(&x)
    })?;

    // 2. Load weights
    let weights = ckpt.load_weights(runner.signature())?;
    let sess = runner.session(weights);

    // 3. Run
    let y = sess.run(|inputs| {
        inputs.set_input("x", vec![1.0; 8])
    })?;

    println!("{}", y);
    Ok(())
}
```

## Key concepts

| Type | Role |
|---|---|
| `Tensor` | Symbolic tensor — records ops into the graph, not a data buffer |
| `TraceCx` | Tracing context — declares inputs, weights, and naming scopes |
| `Device` | Compilation target wrapping a PJRT plugin (CPU, GPU, …) |
| `CompiledModel` | A compiled PJRT executable paired with its parameter signature |
| `Session` | A model with pre-bound weights, ready for repeated inference |
| `Checkpoint` | In-memory safetensors file for weight shapes and data |
| `#[derive(Module)]` | Auto-generates weight tracing for your model structs |

## Built-in layers

- `Linear` — `x @ W^T + bias` (PyTorch weight layout)
- `Embedding` — token-to-vector lookup table
- `RmsNorm` — Root Mean Square Layer Normalization

## Getting started

### Prerequisites

You need a PJRT plugin for your target backend. For CPU on Apple Silicon:

```bash
just download-pjrt
```

This downloads `libpjrt_cpu.dylib` into the project root. Set `PJRT_CPU_PLUGIN` to override the path.

### Run the linear example

```bash
cd examples/linear
uv run make-safetensor.py   # generate dummy weights
cargo run --example linear
```

### Run the SmolLM2 chat example

Running chat will compile the graph if there's no pre-compiled artifact. You'll see this in the logs.
```bash
just download-smollm2                          # download weights + tokenizer
cargo build --release --example smollm2
./target/release/examples/smollm2 chat
```
And then try compiling the model graph and starting the chat 
```bash
./target/release/examples/smollm2 compile         # compile the model graph
./target/release/examples/smollm2 chat --compiled examples/smollm2/artifacts/smollm2.compiled
```

## Debugging

Set `FUSEBOX_DUMP_MLIR=1` to print the generated StableHLO MLIR to stderr before compilation:

```bash
FUSEBOX_DUMP_MLIR=1 cargo run --example linear
```

## Project structure

```
src/
├── lib.rs                  # Crate root and prelude
├── tensor.rs               # Symbolic Tensor API (user-facing)
├── trace.rs                # TraceCx — graph tracing entry point
├── builder.rs              # FuncBuilder — emits IR from tensor ops
├── ir.rs                   # IR data structures (mirrors StableHLO)
├── print_mlir.rs           # MLIR text emitter
├── pjrt_runtime.rs         # CompiledModel, Session, execution via PJRT
├── signature.rs            # Parameter signatures and input binding
├── device.rs               # Device abstraction over PJRT plugins
├── checkpoint.rs           # Safetensors checkpoint loader
├── weights.rs              # Weight extraction with bf16/f16→f32 conversion
├── safetensor_shapes.rs    # Header-only shape parsing from safetensors
├── shape.rs                # Shape (dims + dtype)
├── dtype.rs                # Supported element types
├── value.rs                # SSA value ids
├── error.rs                # Unified error type
├── module_api.rs           # Module and ShapeProvider traits
└── nn/                     # Built-in layers (Linear, Embedding, RmsNorm)

fusebox_macros/         # Proc macro crate (#[derive(Module)])

examples/
├── linear/             # Minimal MLP example
└── smollm2/            # SmolLM2-135M-Instruct transformer with chat CLI
```
