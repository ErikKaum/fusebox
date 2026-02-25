// src/trace.rs
use crate::builder::FuncBuilder;
use crate::error::Error;
use crate::ir::Function;
use crate::shape::Shape;
use crate::tensor::Tensor;

pub struct TraceCx {
    b: FuncBuilder,
    prefix: String,
}

impl TraceCx {
    pub fn new(func_name: impl Into<String>) -> Self {
        Self {
            b: FuncBuilder::new(func_name),
            prefix: String::new(),
        }
    }

    pub fn finish(self) -> Function {
        self.b.finish()
    }

    pub fn ret(&mut self, t: &Tensor) {
        self.b.ret(t)
    }

    pub fn input(&mut self, name: &str, shape: Shape) -> Tensor {
        let full = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}{}", self.prefix, name)
        };
        self.b.input(full, shape)
    }

    pub fn weight(&mut self, name: &str, shape: Shape) -> Tensor {
        let full = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}{}", self.prefix, name)
        };
        self.b.weight(full, shape)
    }

    /// Convert a local name like "w" into a fully-qualified name like "proj/w"
    /// according to the current scope stack.
    pub fn qualify(&self, local: &str) -> String {
        if self.prefix.is_empty() {
            local.to_string()
        } else {
            format!("{}{}", self.prefix, local)
        }
    }

    /// Push a naming scope; returns the prefix length before the push.
    /// Call `pop_scope` with the returned value to restore.
    pub fn push_scope(&mut self, scope: &str) -> usize {
        let prev_len = self.prefix.len();
        if self.prefix.is_empty() {
            self.prefix = format!("{}/", scope);
        } else {
            self.prefix.push_str(scope);
            self.prefix.push('/');
        }
        prev_len
    }

    pub fn pop_scope(&mut self, prev_len: usize) {
        self.prefix.truncate(prev_len);
    }

    pub fn matmul_2d(&mut self, x: &Tensor, w: &Tensor) -> Result<Tensor, Error> {
        self.b.matmul_2d(x, w)
    }

    pub fn broadcast_bias_1d(&mut self, b: &Tensor, batch: i64) -> Result<Tensor, Error> {
        self.b.broadcast_bias_1d(b, batch)
    }

    pub fn add(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
        self.b.add(a, b)
    }

    pub fn mul(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
        self.b.mul(a, b)
    }

    pub fn silu(&mut self, x: &Tensor) -> Result<Tensor, Error> {
        self.b.silu(x)
    }
}

