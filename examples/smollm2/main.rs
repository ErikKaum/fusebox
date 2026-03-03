mod model;

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};
use tokenizers::Tokenizer;

use fusebox::prelude::*;
use model::trace_smollm2;

// ── CLI ─────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "smollm2", about = "SmolLM2-135M: compile and chat")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile the model graph and save the artifact to disk.
    Compile {
        /// Path to the safetensors checkpoint (needed for weight shapes).
        #[arg(long, default_value = "examples/smollm2/artifacts/smollm2-135m.safetensors")]
        checkpoint: PathBuf,

        /// Where to write the compiled artifact.
        #[arg(long, default_value = "examples/smollm2/artifacts/smollm2.compiled")]
        output: PathBuf,

        /// Sequence length baked into the compiled graph.
        #[arg(long, default_value_t = 256)]
        seq: i64,
    },
    /// Interactive chat using the model.
    Chat {
        /// Path to the safetensors checkpoint (weights).
        #[arg(long, default_value = "examples/smollm2/artifacts/smollm2-135m.safetensors")]
        checkpoint: PathBuf,

        /// Path to a HuggingFace tokenizer.json file.
        #[arg(long, default_value = "examples/smollm2/artifacts/tokenizer.json")]
        tokenizer: PathBuf,

        /// Path to a pre-compiled artifact. If omitted, compiles from scratch.
        #[arg(long)] 
        compiled: Option<PathBuf>,

        /// Sequence length (must match the compiled artifact if one is provided).
        #[arg(long, default_value_t = 256)]
        seq: i64,

        /// Maximum number of new tokens to generate per turn.
        #[arg(long, default_value_t = 200)]
        max_tokens: usize,
    },
}

// ── Compile ─────────────────────────────────────────────────────────

fn compile_model(
    checkpoint: &PathBuf,
    output: &PathBuf,
    seq: i64,
) -> Result<(), Box<dyn std::error::Error>> {
    let ckpt = Checkpoint::from_file(checkpoint)?;
    let device = Device::cpu();

    let batch: i64 = 1;

    println!("Tracing + compiling (seq={seq})...");
    let t0 = Instant::now();
    let runner = device.compile("smollm2", |cx| trace_smollm2(cx, ckpt.shapes(), batch, seq))?;
    println!("Compiled in {:.2?}", t0.elapsed());

    runner.save(output)?;
    println!("Saved to {}", output.display());

    Ok(())
}

// ── Chat ────────────────────────────────────────────────────────────

fn chat(
    checkpoint: &PathBuf,
    tokenizer_path: &PathBuf,
    compiled: Option<&PathBuf>,
    seq: i64,
    max_tokens: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let ckpt = Checkpoint::from_file(checkpoint)?;
    let device = Device::cpu();
    let batch: i64 = 1;

    let t0 = Instant::now();
    let runner = match compiled {
        Some(path) => {
            let r = device.load(path)?;
            println!("Loaded compiled model in {:.2?}", t0.elapsed());
            r
        }
        None => {
            println!("No compiled artifact given, compiling from scratch (seq={seq})...");
            let r =
                device.compile("smollm2", |cx| trace_smollm2(cx, ckpt.shapes(), batch, seq))?;
            println!("Compiled in {:.2?}", t0.elapsed());
            r
        }
    };

    let t0 = Instant::now();
    let weights = ckpt.load_weights(runner.signature())?;
    println!("Loaded weights in {:.2?}", t0.elapsed());

    let sess = runner.session(weights);

    let tokenizer =
        Tokenizer::from_file(tokenizer_path).map_err(|e| format!("load tokenizer: {e}"))?;

    let eos_id = tokenizer
        .token_to_id("<|endoftext|>")
        .or_else(|| tokenizer.token_to_id("</s>"))
        .unwrap_or(2) as i32;

    println!("\nReady! Type a prompt and press Enter. Type \"exit\" to quit.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush()?;

        let mut prompt = String::new();
        if stdin.lock().read_line(&mut prompt)? == 0 {
            break; // EOF
        }
        let prompt = prompt.trim();
        if prompt.is_empty() {
            continue;
        }
        if prompt == "exit" || prompt == "quit" {
            break;
        }

        let encoding = tokenizer
            .encode(prompt, false)
            .map_err(|e| format!("tokenize: {e}"))?;
        let prompt_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();

        let mut context = prompt_ids;

        for _ in 0..max_tokens {
            let input = prepare_input(&context, seq as usize);

            let result = sess.run(|inputs| inputs.set_input_i32("tokens", input))?;

            let next_token = result.to_i32().unwrap()[0];
            if next_token == eos_id {
                break;
            }

            context.push(next_token);

            let piece = tokenizer
                .decode(&[next_token as u32], true)
                .unwrap_or_default();
            print!("{piece}");
            stdout.flush()?;
        }

        println!();
    }

    Ok(())
}

/// Build a [seq]-length i32 input from the current context.
/// Left-pads with 0 if context is shorter than seq, or takes the last
/// `seq` tokens when context has grown past the window.
fn prepare_input(context: &[i32], seq: usize) -> Vec<i32> {
    if context.len() >= seq {
        context[context.len() - seq..].to_vec()
    } else {
        let mut padded = vec![0i32; seq - context.len()];
        padded.extend_from_slice(context);
        padded
    }
}

// ── Main ────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match &cli.command {
        Command::Compile {
            checkpoint,
            output,
            seq,
        } => compile_model(checkpoint, output, *seq),
        Command::Chat {
            checkpoint,
            tokenizer,
            compiled,
            seq,
            max_tokens,
        } => chat(checkpoint, tokenizer, compiled.as_ref(), *seq, *max_tokens),
    }
}
