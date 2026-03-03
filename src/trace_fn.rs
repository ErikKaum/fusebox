use crate::error::Error;
use crate::ir::Function;
use crate::tensor::Tensor;
use crate::trace::TraceCx;

pub fn trace_function<F>(name: &str, build: F) -> Result<Function, Error>
where
    F: FnOnce(&mut TraceCx) -> Result<Tensor, Error>,
{
    let mut cx = TraceCx::new(name);
    let out = build(&mut cx)?;
    cx.set_ret(&out);
    drop(out);
    Ok(cx.finish())
}
