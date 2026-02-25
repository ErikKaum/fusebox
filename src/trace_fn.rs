use crate::{error::Error, ir::Function, tensor::Tensor, trace::TraceCx};

pub fn trace_function<F>(name: &str, build: F) -> Result<Function, Error>
where
    F: FnOnce(&mut TraceCx) -> Result<Tensor, Error>,
{
    let mut cx = TraceCx::new(name);
    let out = build(&mut cx)?;
    cx.ret(&out);
    Ok(cx.finish())
}