// src/pjrt_runtime.rs (additions / edits)
use std::path::{Path, PathBuf};
use std::sync::Arc;

use pjrt::{Client, F32, HostBuffer, LoadedExecutable, Program, ProgramFormat, TypedHostBuffer};

use crate::ir::{Function, Module};
use crate::print_mlir::print_module;
use crate::shape::Shape;
use crate::signature::{InputsF32, Signature};

#[derive(Debug, Clone)]
pub struct HostTensorF32 {
    pub dims: Vec<i64>,
    pub data: Vec<f32>,
}

impl HostTensorF32 {
    pub fn from_shape(shape: &Shape, data: Vec<f32>) -> Result<Self, String> {
        let want = shape
            .dims
            .iter()
            .copied()
            .map(|d| d as i128)
            .product::<i128>()
            .max(1);
        let got = data.len() as i128;
        if want != got {
            return Err(format!(
                "shape {:?} implies {} elements, got {}",
                shape.dims, want, got
            ));
        }
        Ok(Self {
            dims: shape.dims.clone(),
            data,
        })
    }
}

pub struct PjrtCpuRunner {
    client: Client,
    exec: LoadedExecutable,
    sig: Arc<Signature>,
}

impl PjrtCpuRunner {
    // New: build from your traced Function (best ergonomics).
    pub fn from_function(func: &Function, plugin_path: impl AsRef<Path>) -> Result<Self, String> {
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
    ) -> Result<Self, String> {
        let plugin_path = plugin_path.as_ref();

        let plugin_str = plugin_path
            .to_str()
            .ok_or_else(|| format!("plugin path {:?} is not valid UTF-8", plugin_path))?;

        let api = pjrt::plugin(plugin_str)
            .load()
            .map_err(|e| format!("load PJRT plugin {:?}: {}", plugin_path, e))?;

        let client = Client::builder(&api)
            .build()
            .map_err(|e| format!("create PJRT client: {}", e))?;

        let program = Program::new(ProgramFormat::MLIR, mlir.as_bytes().to_vec());
        let exec = LoadedExecutable::builder(&client, &program)
            .build()
            .map_err(|e| format!("compile MLIR to executable: {}", e))?;

        Ok(Self {
            client,
            exec,
            sig: Arc::new(sig),
        })
    }

    pub fn inputs_f32(&self) -> InputsF32 {
        InputsF32::new(self.sig.clone())
    }

    pub fn signature(&self) -> &Signature {
        &self.sig
    }

    pub fn run_f32_inputs(&self, inputs: InputsF32) -> Result<HostTensorF32, String> {
        let ordered = inputs.into_ordered()?; // Vec<(Shape, Vec<f32>)> in param order

        let mut device_inputs = Vec::with_capacity(ordered.len());
        for (shape, data) in ordered {
            let typed = TypedHostBuffer::<F32>::builder()
                .data::<f32>(data)
                .maybe_dims(Some(shape.dims))
                .build();

            let host: HostBuffer = HostBuffer::from(typed);
            let buf = host
                .copy_to_sync(&self.client)
                .map_err(|e| format!("copy input to device: {}", e))?;

            device_inputs.push(buf);
        }

        let results = self
            .exec
            .execution(device_inputs)
            .run_sync()
            .map_err(|e| format!("execute: {}", e))?;

        let out_buf = results
            .get(0)
            .and_then(|xs| xs.get(0))
            .ok_or_else(|| "no outputs returned".to_string())?;

        let host_out = out_buf
            .copy_to_host_sync()
            .map_err(|e| format!("copy output to host: {}", e))?;

        match host_out {
            HostBuffer::F32(tb) => Ok(HostTensorF32 {
                dims: tb.dims().to_vec(),
                data: tb.data().to_vec(),
            }),
            other => Err(format!("expected f32 output, got {:?}", other)),
        }
    }

    pub fn client(&self) -> &Client {
        &self.client
    }
}

// unchanged helper
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
