extern crate self as fusebox;

pub mod builder;
pub mod checkpoint;
pub mod device;
pub mod dtype;
pub mod error;
pub mod ir;
pub mod module_api;
pub mod nn;
pub mod pjrt_runtime;
pub mod print_mlir;
pub mod safetensor_shapes;
pub mod shape;
pub mod signature;
pub mod tensor;
pub mod trace;
pub mod trace_fn;
pub mod value;
pub mod weights;

pub mod prelude {
    pub use crate::checkpoint::Checkpoint;
    pub use crate::device::Device;
    pub use crate::dtype::DType;
    pub use crate::error::Error;
    pub use crate::module_api::{Module, ShapeProvider};
    pub use crate::nn::{Embedding, Linear, RmsNorm};
    pub use crate::pjrt_runtime::{CompiledModel, HostTensor, TensorData};
    pub use crate::shape::Shape;
    pub use crate::tensor::Tensor;
    pub use crate::trace::TraceCx;
    pub use fusebox_macros::Module;
}
