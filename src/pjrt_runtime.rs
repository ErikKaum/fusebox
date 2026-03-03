use std::path::{Path, PathBuf};
use std::sync::Arc;

use pjrt::{
    Buffer, Client, F32, HostBuffer, LoadedExecutable, Program, ProgramFormat, TypedHostBuffer,
};

use crate::error::Error;
use crate::ir::{Function, Module};
use crate::print_mlir::print_module;
use crate::signature::{Inputs, Signature};
use crate::weights::Weights;

#[derive(Debug, Clone)]
pub struct HostTensor {
    pub dims: Vec<i64>,
    pub data: Vec<f32>,
}

pub struct CompiledModel {
    client: Client,
    exec: LoadedExecutable,
    sig: Arc<Signature>,
}

pub struct Session<'a> {
    runner: &'a CompiledModel,
    weights: Weights,
}

impl CompiledModel {
    pub fn from_function(func: &Function, plugin_path: impl AsRef<Path>) -> Result<Self, Error> {
        let module = Module {
            functions: vec![func.clone()],
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

    pub fn run(&self, inputs: Inputs) -> Result<HostTensor, Error> {
        let ordered = inputs.into_ordered()?;

        let mut device_inputs = Vec::with_capacity(ordered.len());
        for (shape, data) in ordered {
            let buf = f32_host_buffer(&data, &shape.dims)
                .copy_to_sync(&self.client)
                .map_err(|e| Error::RuntimeError(format!("copy input to device: {}", e)))?;
            device_inputs.push(buf);
        }

        execute_and_extract_f32(&self.exec, device_inputs)
    }
}

impl<'a> Session<'a> {
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

fn f32_host_buffer(data: &[f32], dims: &[i64]) -> HostBuffer {
    let typed = TypedHostBuffer::<F32>::builder()
        .data::<f32>(data.to_vec())
        .maybe_dims(Some(dims.to_vec()))
        .build();
    HostBuffer::from(typed)
}

fn execute_and_extract_f32(
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
            data: tb.data().to_vec(),
        }),
        other => Err(Error::RuntimeError(format!(
            "expected f32 output, got {:?}",
            other
        ))),
    }
}

pub fn default_cpu_plugin_path() -> PathBuf {
    if let Ok(p) = std::env::var("PJRT_CPU_PLUGIN") {
        return PathBuf::from(p);
    }
    let dylib = PathBuf::from("libpjrt_cpu.dylib");
    if dylib.exists() {
        return dylib;
    }
    PathBuf::from("pjrt_c_api_cpu_plugin.so")
}
