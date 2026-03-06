//! PJRT-backed runtime: compile MLIR programs and execute them on hardware.
//!
//! The main types here are:
//! - [`CompiledModel`] — a compiled PJRT executable paired with its parameter signature.
//! - [`Session`] — a model + pre-bound weights, ready for repeated inference.
//! - [`HostTensor`] / [`TensorData`] — concrete host-side tensor data returned from execution.

use core::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use pjrt::{
    Buffer, Client, F32, HostBuffer, I32, LoadedExecutable, Program, ProgramFormat, TypedHostBuffer,
};

use crate::dtype::DType;
use crate::error::Error;
use crate::ir::{Function, Module};
use crate::print_mlir::print_module;
use crate::signature::{Inputs, ParamData, Signature};
use crate::weights::Weights;

// ── TensorData ──────────────────────────────────────────────────────

/// Type-erased tensor payload (the raw element data without shape).
#[derive(Debug, Clone)]
pub enum TensorData {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

impl TensorData {
    pub fn dtype(&self) -> DType {
        match self {
            TensorData::F32(_) => DType::F32,
            TensorData::I32(_) => DType::I32,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TensorData::F32(v) => v.len(),
            TensorData::I32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            TensorData::F32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<&[i32]> {
        match self {
            TensorData::I32(v) => Some(v),
            _ => None,
        }
    }
}

// ── HostTensor ──────────────────────────────────────────────────────

/// A concrete tensor living in host (CPU) memory — the result you get back
/// after running a compiled model.
#[derive(Debug, Clone)]
pub struct HostTensor {
    pub dims: Vec<i64>,
    pub data: TensorData,
}

impl HostTensor {
    pub fn shape(&self) -> &[i64] {
        &self.dims
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    pub fn to_f32(&self) -> Option<&[f32]> {
        self.data.as_f32()
    }

    pub fn to_i32(&self) -> Option<&[i32]> {
        self.data.as_i32()
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>, Error> {
        self.to_f32()
            .map(|s| s.to_vec())
            .ok_or_else(|| Error::RuntimeError(format!("expected f32, got {:?}", self.dtype())))
    }
}

impl fmt::Display for HostTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape_str: Vec<String> = self.dims.iter().map(|d| d.to_string()).collect();
        write!(
            f,
            "HostTensor([{}], {}, {} elements)",
            shape_str.join(", "),
            self.dtype(),
            self.numel()
        )
    }
}

// ── CompiledModel ───────────────────────────────────────────────────

/// A compiled PJRT executable + its parameter signature.
///
/// Created via [`Device::compile`](crate::device::Device::compile) or
/// [`CompiledModel::from_function`]. Holds a PJRT client and loaded
/// executable; call [`run`](Self::run) or create a [`Session`] for inference.
pub struct CompiledModel {
    client: Client,
    exec: LoadedExecutable,
    sig: Arc<Signature>,
}

/// A model with pre-bound weights, ready for repeated inference calls
/// where only the inputs change between runs.
pub struct Session<'a> {
    runner: &'a CompiledModel,
    weights: Weights,
}

impl CompiledModel {
    /// Compile a traced [`Function`] into a PJRT executable.
    pub fn from_function(func: &Function, plugin_path: impl AsRef<Path>) -> Result<Self, Error> {
        let mut main_func = func.clone();
        main_func.name = "main".to_string();
        let module = Module {
            functions: vec![main_func],
        };
        let mlir = print_module(&module);
        Self::from_mlir_text_and_sig(&mlir, Signature::from_function(func), plugin_path)
    }

    fn from_mlir_text_and_sig(
        mlir: &str,
        sig: Signature,
        plugin_path: impl AsRef<Path>,
    ) -> Result<Self, Error> {
        let plugin_path = plugin_path.as_ref();

        let plugin_str = plugin_path.to_str().ok_or_else(|| {
            Error::CompilationError(format!("plugin path {:?} is not valid UTF-8", plugin_path))
        })?;

        let api = pjrt::plugin(plugin_str).load().map_err(|e| {
            Error::CompilationError(format!("load PJRT plugin {:?}: {}", plugin_path, e))
        })?;

        let client = Client::builder(&api)
            .build()
            .map_err(|e| Error::CompilationError(format!("create PJRT client: {}", e)))?;

        let program = Program::new(ProgramFormat::MLIR, mlir.as_bytes().to_vec());
        let exec = LoadedExecutable::builder(&client, &program)
            .build()
            .map_err(|e| Error::CompilationError(format!("compile MLIR to executable: {}", e)))?;

        Ok(Self {
            client,
            exec,
            sig: Arc::new(sig),
        })
    }

    pub fn inputs(&self) -> Inputs {
        Inputs::new(self.sig.clone())
    }

    pub fn signature(&self) -> &Signature {
        &self.sig
    }

    pub fn session(&self, weights: Weights) -> Session<'_> {
        Session {
            runner: self,
            weights,
        }
    }

    /// Execute the model with the given inputs, copying data to/from the device.
    pub fn run(&self, inputs: Inputs) -> Result<HostTensor, Error> {
        let ordered = inputs.into_ordered()?;

        let mut device_inputs = Vec::with_capacity(ordered.len());
        for (shape, data) in ordered {
            let host_buf = match data {
                ParamData::F32(v) => make_f32_host_buffer(&v, &shape.dims),
                ParamData::I32(v) => make_i32_host_buffer(&v, &shape.dims),
            };
            let buf = host_buf
                .copy_to_sync(&self.client)
                .map_err(|e| Error::RuntimeError(format!("copy input to device: {}", e)))?;
            device_inputs.push(buf);
        }

        execute_and_extract(&self.exec, device_inputs)
    }

    /// Serialize the compiled executable and signature to a file.
    ///
    /// Format: [8-byte LE sig length] [signature JSON] [PJRT executable bytes]
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let sig_bytes = serde_json::to_vec(&*self.sig)
            .map_err(|e| Error::SerializationError(format!("serialize signature: {}", e)))?;

        let executable = self.exec.executable();
        let serialized = executable.serialize();
        let exec_bytes = serialized.bytes();

        let mut file = std::fs::File::create(path)?;
        file.write_all(&(sig_bytes.len() as u64).to_le_bytes())?;
        file.write_all(&sig_bytes)?;
        file.write_all(exec_bytes)?;

        Ok(())
    }

    /// Load a previously saved compiled model from bytes.
    ///
    /// The `client` must use the same PJRT backend that was used to compile.
    pub(crate) fn from_bytes(client: Client, data: &[u8]) -> Result<Self, Error> {
        if data.len() < 8 {
            return Err(Error::SerializationError(
                "file too small to contain header".to_string(),
            ));
        }

        let sig_len =
            u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;

        if data.len() < 8 + sig_len {
            return Err(Error::SerializationError(
                "file truncated: signature data missing".to_string(),
            ));
        }

        let sig: Signature = serde_json::from_slice(&data[8..8 + sig_len])
            .map_err(|e| Error::SerializationError(format!("deserialize signature: {}", e)))?;

        let exec_bytes = &data[8 + sig_len..];
        let exec = client
            .load_executable(exec_bytes)
            .map_err(|e| Error::SerializationError(format!("load executable: {}", e)))?;

        Ok(Self {
            client,
            exec,
            sig: Arc::new(sig),
        })
    }
}

impl Session<'_> {
    pub fn run(
        &self,
        set_inputs: impl FnOnce(&mut Inputs) -> Result<(), Error>,
    ) -> Result<HostTensor, Error> {
        let mut inputs = self.runner.inputs();
        self.weights.apply_ref(&mut inputs)?;
        set_inputs(&mut inputs)?;
        self.runner.run(inputs)
    }
}

// ── Buffer helpers ──────────────────────────────────────────────────

fn make_f32_host_buffer(data: &[f32], dims: &[i64]) -> HostBuffer {
    let typed = TypedHostBuffer::<F32>::builder()
        .data::<f32>(data.to_vec())
        .maybe_dims(Some(dims.to_vec()))
        .build();
    HostBuffer::from(typed)
}

fn make_i32_host_buffer(data: &[i32], dims: &[i64]) -> HostBuffer {
    let typed = TypedHostBuffer::<I32>::builder()
        .data::<i32>(data.to_vec())
        .maybe_dims(Some(dims.to_vec()))
        .build();
    HostBuffer::from(typed)
}

/// Run the executable and copy the first output back to host memory.
fn execute_and_extract(
    exec: &LoadedExecutable,
    arg_buffers: Vec<Buffer>,
) -> Result<HostTensor, Error> {
    let results = exec
        .execution(arg_buffers)
        .run_sync()
        .map_err(|e| Error::RuntimeError(format!("execute: {}", e)))?;

    let out_buf = results
        .first()
        .and_then(|xs| xs.first())
        .ok_or_else(|| Error::RuntimeError("no outputs returned".to_string()))?;

    let host_out = out_buf
        .copy_to_host_sync()
        .map_err(|e| Error::RuntimeError(format!("copy output to host: {}", e)))?;

    match host_out {
        HostBuffer::F32(tb) => Ok(HostTensor {
            dims: tb.dims().to_vec(),
            data: TensorData::F32(tb.data().to_vec()),
        }),
        HostBuffer::I32(tb) => Ok(HostTensor {
            dims: tb.dims().to_vec(),
            data: TensorData::I32(tb.data().to_vec()),
        }),
        other => Err(Error::RuntimeError(format!(
            "unsupported output type: {:?}",
            other
        ))),
    }
}

/// Resolve the CPU PJRT plugin path: check `PJRT_CPU_PLUGIN` env var,
/// then fall back to platform-specific default names.
pub fn default_cpu_plugin_path() -> PathBuf {
    if let Ok(p) = std::env::var("PJRT_CPU_PLUGIN") {
        return PathBuf::from(p);
    }
    let dylib = PathBuf::from("libpjrt_cpu.dylib");
    if dylib.exists() {
        return dylib;
    }
    PathBuf::from("libpjrt_cpu.so")
}
