//! Tracing context — the entry point for building a computation graph.
//!
//! Create a [`TraceCx`], declare inputs and weights, perform tensor operations,
//! then call [`TraceCx::finish`] to extract the completed [`Function`] IR.

use std::cell::RefCell;
use std::rc::Rc;

use crate::builder::FuncBuilder;
use crate::error::Error;
use crate::ir::Function;
use crate::shape::Shape;
use crate::tensor::Tensor;

/// Tracing context for building computation graphs.
///
/// Used during model tracing to create input/weight parameters and manage
/// hierarchical name scoping. Operations themselves live on [`Tensor`] —
/// you no longer need to thread `TraceCx` through forward methods.
pub struct TraceCx {
    builder: Rc<RefCell<FuncBuilder>>,
    prefix: String,
}

impl TraceCx {
    pub fn new(func_name: impl Into<String>) -> Self {
        Self {
            builder: Rc::new(RefCell::new(FuncBuilder::new(func_name))),
            prefix: String::new(),
        }
    }

    /// Mark a tensor as the function's return value.
    pub fn set_ret(&mut self, t: &Tensor) {
        self.builder.borrow_mut().set_return(t.value);
    }

    /// Consume the tracing context and return the completed IR function.
    pub fn finish(self) -> Function {
        match Rc::try_unwrap(self.builder) {
            Ok(cell) => cell.into_inner().into_function(),
            Err(rc) => rc.borrow().function().clone(),
        }
    }

    /// Declare a runtime input parameter (e.g. token ids, pixel data).
    pub fn input(&mut self, name: &str, shape: Shape) -> Tensor {
        let full = self.qualify(name);
        let mut b = self.builder.borrow_mut();
        let id = b.add_input(full, shape.clone());
        drop(b);
        Tensor::new(shape, id, &self.builder)
    }

    /// Declare a weight parameter (loaded from a checkpoint at runtime).
    pub fn weight(&mut self, name: &str, shape: Shape) -> Tensor {
        let full = self.qualify(name);
        let mut b = self.builder.borrow_mut();
        let id = b.add_weight(full, shape.clone());
        drop(b);
        Tensor::new(shape, id, &self.builder)
    }

    /// Prepend the current scope prefix to a local name (e.g. "layers/0/attn/weight").
    pub fn qualify(&self, local: &str) -> String {
        if self.prefix.is_empty() {
            local.to_string()
        } else {
            format!("{}{}", self.prefix, local)
        }
    }

    /// Push a hierarchical naming scope. Returns the previous prefix length
    /// so it can be restored via [`pop_scope`](Self::pop_scope).
    pub fn push_scope(&mut self, scope: &str) -> usize {
        let prev_len = self.prefix.len();
        if self.prefix.is_empty() {
            self.prefix = format!("{}/", scope);
        } else {
            self.prefix.push_str(scope);
            self.prefix.push('/');
        }
        self.builder.borrow_mut().push_scope(scope);
        prev_len
    }

    pub fn pop_scope(&mut self, prev_len: usize) {
        self.prefix.truncate(prev_len);
        self.builder.borrow_mut().pop_scope();
    }

    /// Create an iota tensor (sequential indices along `dimension`).
    pub fn iota(&mut self, shape: Shape, dimension: i64) -> Result<Tensor, Error> {
        let mut b = self.builder.borrow_mut();
        let (out_shape, value) = b.iota(&shape, dimension)?;
        drop(b);
        Ok(Tensor::new(out_shape, value, &self.builder))
    }
}
