use std::path::PathBuf;

use crate::error::Error;
use crate::pjrt_runtime::{CompiledModel, default_cpu_plugin_path};
use crate::tensor::Tensor;
use crate::trace::TraceCx;
use crate::trace_fn::trace_function;

/// A target device for compilation and execution.
///
/// Wraps a PJRT plugin. Use [`Device::cpu()`] for the default CPU backend,
/// or [`Device::from_plugin`] to load a custom PJRT plugin (e.g. GPU).
pub struct Device {
    plugin_path: PathBuf,
}

impl Device {
    pub fn cpu() -> Self {
        Device {
            plugin_path: default_cpu_plugin_path(),
        }
    }

    pub fn from_plugin(path: impl Into<PathBuf>) -> Self {
        Device {
            plugin_path: path.into(),
        }
    }

    /// Trace and compile a computation graph in one step.
    ///
    /// The closure receives a [`TraceCx`] for declaring inputs and weights.
    /// Operations are expressed as method calls on [`Tensor`].
    pub fn compile<F>(&self, name: &str, build: F) -> Result<CompiledModel, Error>
    where
        F: FnOnce(&mut TraceCx) -> Result<Tensor, Error>,
    {
        let func = trace_function(name, build)?;
        if std::env::var("FUSEBOX_DUMP_MLIR").is_ok() {
            let module = crate::ir::Module {
                functions: vec![func.clone()],
            };
            eprintln!("{}", crate::print_mlir::print_module(&module));
        }
        CompiledModel::from_function(&func, &self.plugin_path)
    }
}
