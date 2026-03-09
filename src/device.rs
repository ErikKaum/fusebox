use std::path::{Path, PathBuf};

use pjrt::Client;

use crate::error::Error;
use crate::ir::{Function, ParamKind};
use crate::module_api::ShapeProvider;
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
    pub fn cpu() -> Result<Self, Error> {
        Ok(Device {
            plugin_path: default_cpu_plugin_path()?,
        })
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

    /// Load a previously saved compiled model from disk.
    ///
    /// The model must have been compiled for the same backend (e.g. CPU plugin)
    /// that this `Device` is configured with.
    pub fn load(&self, path: impl AsRef<Path>) -> Result<CompiledModel, Error> {
        let data = std::fs::read(path.as_ref())?;
        let client = self.make_client()?;
        CompiledModel::from_bytes(client, &data)
    }

    fn make_client(&self) -> Result<Client, Error> {
        let plugin_str = self.plugin_path.to_str().ok_or_else(|| {
            Error::CompilationError(format!(
                "plugin path {:?} is not valid UTF-8",
                self.plugin_path
            ))
        })?;

        let api = pjrt::plugin(plugin_str).load().map_err(|e| {
            Error::CompilationError(format!("load PJRT plugin {:?}: {}", self.plugin_path, e))
        })?;

        Client::builder(&api)
            .build()
            .map_err(|e| Error::CompilationError(format!("create PJRT client: {}", e)))
    }
}

/// Validate that a checkpoint's shapes match all weight parameters in a traced function.
///
/// Returns `Ok(())` if every weight param has a matching shape in the checkpoint.
/// Returns an error listing all mismatches at once rather than failing on the first one.
pub fn validate_weights(func: &Function, shapes: &dyn ShapeProvider) -> Result<(), Error> {
    let mut errors = Vec::new();

    for p in &func.params {
        if p.kind != ParamKind::Weight {
            continue;
        }
        match shapes.shape_of(&p.name) {
            Ok(Some(file_shape)) => {
                if file_shape.dims != p.shape.dims {
                    errors.push(format!(
                        "  {:?}: expected shape {:?}, checkpoint has {:?}",
                        p.name, p.shape.dims, file_shape.dims
                    ));
                }
                if file_shape.dtype != p.shape.dtype {
                    errors.push(format!(
                        "  {:?}: expected dtype {}, checkpoint has {}",
                        p.name, p.shape.dtype, file_shape.dtype
                    ));
                }
            }
            Ok(None) => {
                errors.push(format!("  {:?}: missing from checkpoint", p.name));
            }
            Err(e) => {
                errors.push(format!("  {:?}: error reading shape: {}", p.name, e));
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(Error::ValidationError(format!(
            "weight validation failed ({} issue{}):\n{}",
            errors.len(),
            if errors.len() == 1 { "" } else { "s" },
            errors.join("\n")
        )))
    }
}
