// src/pjrt_runtime.rs
//
// Minimal PJRT runner for StableHLO MLIR (CPU plugin), f32-only for now.
// Expects the MLIR module to contain `func.func @main(...) -> ...`
// (same requirement you hit in the Go runner).

use std::path::{Path, PathBuf};

use pjrt::{Client, F32, HostBuffer, LoadedExecutable, Program, ProgramFormat, TypedHostBuffer};

#[derive(Debug, Clone)]
pub struct HostTensorF32 {
    pub dims: Vec<i64>,
    pub data: Vec<f32>,
}

impl HostTensorF32 {
    pub fn new(dims: Vec<i64>, data: Vec<f32>) -> Result<Self, String> {
        let want = dims.iter().copied().map(|d| d as i128).product::<i128>();
        let got = data.len() as i128;
        if want != got {
            return Err(format!(
                "shape {:?} implies {} elements, got {}",
                dims, want, got
            ));
        }
        Ok(Self { dims, data })
    }
}

pub struct PjrtCpuRunner {
    client: Client,
    exec: LoadedExecutable,
}

impl PjrtCpuRunner {
    pub fn from_mlir_text(mlir: &str, plugin_path: impl AsRef<Path>) -> Result<Self, String> {
        let plugin_path = plugin_path.as_ref();

        // Load PJRT C API from the plugin dynamic library.
        let plugin_str = plugin_path
            .to_str()
            .ok_or_else(|| format!("plugin path {:?} is not valid UTF-8", plugin_path))?;

        let api = pjrt::plugin(plugin_str)
            .load()
            .map_err(|e| format!("load PJRT plugin {:?}: {}", plugin_path, e))?;

        let client = Client::builder(&api)
            .build()
            .map_err(|e| format!("create PJRT client: {}", e))?;

        // Compile the MLIR (StableHLO) module into a loaded executable.
        let program = Program::new(ProgramFormat::MLIR, mlir.as_bytes().to_vec());
        let exec = LoadedExecutable::builder(&client, &program)
            .build()
            .map_err(|e| format!("compile MLIR to executable: {}", e))?;

        Ok(Self { client, exec })
    }

    pub fn run_f32(&self, inputs: Vec<HostTensorF32>) -> Result<HostTensorF32, String> {
        // Copy inputs to device buffers.
        // For now: f32-only, row-major dense.
        let mut device_inputs = Vec::with_capacity(inputs.len());
        for t in inputs {
            let typed = TypedHostBuffer::<F32>::builder()
                .data::<f32>(t.data)
                .maybe_dims(Some(t.dims))
                .build();

            let host: HostBuffer = HostBuffer::from(typed);
            let buf = host
                .copy_to_sync(&self.client)
                .map_err(|e| format!("copy input to device: {}", e))?;

            device_inputs.push(buf);
        }

        // Execute (single device, single replica is the default for CPU plugin).
        let results = self
            .exec
            .execution(device_inputs)
            .run_sync()
            .map_err(|e| format!("execute: {}", e))?;

        // results[device_index][output_index]
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

// Small helper: choose plugin path from env, or fall back to a local filename.
pub fn default_cpu_plugin_path() -> PathBuf {
    if let Ok(p) = std::env::var("PJRT_CPU_PLUGIN") {
        return PathBuf::from(p);
    }
    // macOS tends to be .dylib, Linux .so. Use whichever you have.
    let dylib = PathBuf::from("libpjrt_cpu.dylib");
    if dylib.exists() {
        return dylib;
    }
    PathBuf::from("pjrt_c_api_cpu_plugin.so")
}
