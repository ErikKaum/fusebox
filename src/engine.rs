// src/engine.rs
use std::path::{Path, PathBuf};

use crate::{
    error::Error,
    ir::Function,
    pjrt_runtime::PjrtCpuRunner,
    tensor::Tensor,
    trace::TraceCx,
    trace_fn::trace_function,
};

pub struct Engine {
    plugin_path: PathBuf,
}

impl Engine {
    pub fn new(plugin_path: impl AsRef<Path>) -> Self {
        Self { plugin_path: plugin_path.as_ref().to_path_buf() }
    }

    pub fn compile<F>(&self, func_name: &str, build: F) -> Result<PjrtCpuRunner, String>
    where
        F: FnOnce(&mut TraceCx) -> Result<Tensor, Error>,
    {
        let func: Function = trace_function(func_name, build).map_err(|e| e.to_string())?;
        PjrtCpuRunner::from_function(&func, &self.plugin_path)
    }
}