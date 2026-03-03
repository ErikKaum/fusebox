use fusebox::dtype::DType;
use fusebox::shape::Shape;
use fusebox::tensor::Tensor;
use fusebox::trace::TraceCx;

fn f32_shape(dims: &[i64]) -> Shape {
    Shape::new(dims.to_vec(), DType::F32)
}

fn trace_with<F>(f: F)
where
    F: FnOnce(&mut TraceCx),
{
    let mut cx = TraceCx::new("test");
    f(&mut cx);
}

// ── Broadcasting ────────────────────────────────────────────────────

#[test]
fn broadcast_scalar_to_vector() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[4]));
        let b = cx.input("b", f32_shape(&[]));
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape.dims, vec![4]);
    });
}

#[test]
fn broadcast_1d_to_2d() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = cx.input("b", f32_shape(&[4]));
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape.dims, vec![3, 4]);
    });
}

#[test]
fn broadcast_expand_ones() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 1]));
        let b = cx.input("b", f32_shape(&[1, 4]));
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape.dims, vec![3, 4]);
    });
}

#[test]
fn broadcast_3d() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[2, 3, 4]));
        let b = cx.input("b", f32_shape(&[4]));
        let c = a.mul(&b).unwrap();
        assert_eq!(c.shape.dims, vec![2, 3, 4]);
    });
}

#[test]
fn broadcast_incompatible_fails() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = cx.input("b", f32_shape(&[5]));
        assert!(a.add(&b).is_err());
    });
}

// ── Matmul ──────────────────────────────────────────────────────────

#[test]
fn matmul_2d() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = cx.input("b", f32_shape(&[4, 5]));
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape.dims, vec![3, 5]);
    });
}

#[test]
fn matmul_batched() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[2, 3, 4]));
        let b = cx.input("b", f32_shape(&[2, 4, 5]));
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape.dims, vec![2, 3, 5]);
    });
}

#[test]
fn matmul_broadcast_weight() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[2, 3, 4]));
        let b = cx.input("b", f32_shape(&[4, 5]));
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape.dims, vec![2, 3, 5]);
    });
}

#[test]
fn matmul_dim_mismatch_fails() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = cx.input("b", f32_shape(&[5, 6]));
        assert!(a.matmul(&b).is_err());
    });
}

// ── Reductions ──────────────────────────────────────────────────────

#[test]
fn reduce_sum_single_axis() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4, 5]));
        let s = a.sum(&[1]).unwrap();
        assert_eq!(s.shape.dims, vec![3, 5]);
    });
}

#[test]
fn reduce_sum_multiple_axes() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4, 5]));
        let s = a.sum(&[0, 2]).unwrap();
        assert_eq!(s.shape.dims, vec![4]);
    });
}

#[test]
fn reduce_sum_negative_axis() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4, 5]));
        let s = a.sum(&[-1]).unwrap();
        assert_eq!(s.shape.dims, vec![3, 4]);
    });
}

#[test]
fn reduce_keepdim() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4, 5]));
        let s = a.sum_keepdim(&[1]).unwrap();
        assert_eq!(s.shape.dims, vec![3, 1, 5]);
    });
}

#[test]
fn reduce_keepdim_multiple_axes() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4, 5]));
        let s = a.sum_keepdim(&[0, 2]).unwrap();
        assert_eq!(s.shape.dims, vec![1, 4, 1]);
    });
}

#[test]
fn mean_shape() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4, 5]));
        let m = a.mean(&[-1]).unwrap();
        assert_eq!(m.shape.dims, vec![3, 4]);
    });
}

// ── Reshape ─────────────────────────────────────────────────────────

#[test]
fn reshape_basic() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = a.reshape(&[12]).unwrap();
        assert_eq!(b.shape.dims, vec![12]);
    });
}

#[test]
fn reshape_element_count_mismatch_fails() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        assert!(a.reshape(&[10]).is_err());
    });
}

// ── Transpose ───────────────────────────────────────────────────────

#[test]
fn transpose_2d() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = a.transpose(&[1, 0]).unwrap();
        assert_eq!(b.shape.dims, vec![4, 3]);
    });
}

#[test]
fn transpose_4d() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[2, 3, 4, 5]));
        let b = a.transpose(&[0, 2, 1, 3]).unwrap();
        assert_eq!(b.shape.dims, vec![2, 4, 3, 5]);
    });
}

// ── Squeeze / Unsqueeze ─────────────────────────────────────────────

#[test]
fn unsqueeze_front() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = a.unsqueeze(0).unwrap();
        assert_eq!(b.shape.dims, vec![1, 3, 4]);
    });
}

#[test]
fn unsqueeze_back() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = a.unsqueeze(-1).unwrap();
        assert_eq!(b.shape.dims, vec![3, 4, 1]);
    });
}

#[test]
fn squeeze_middle() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 1, 4]));
        let b = a.squeeze(1).unwrap();
        assert_eq!(b.shape.dims, vec![3, 4]);
    });
}

// ── Concatenate ─────────────────────────────────────────────────────

#[test]
fn cat_along_axis_0() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[2, 4]));
        let b = cx.input("b", f32_shape(&[3, 4]));
        let c = Tensor::cat(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape.dims, vec![5, 4]);
    });
}

#[test]
fn cat_along_axis_1() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[2, 3]));
        let b = cx.input("b", f32_shape(&[2, 5]));
        let c = Tensor::cat(&[&a, &b], 1).unwrap();
        assert_eq!(c.shape.dims, vec![2, 8]);
    });
}

#[test]
fn cat_dim_mismatch_fails() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[2, 3]));
        let b = cx.input("b", f32_shape(&[4, 5]));
        assert!(Tensor::cat(&[&a, &b], 0).is_err());
    });
}

// ── Slice ───────────────────────────────────────────────────────────

#[test]
fn slice_basic() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[4, 8]));
        let b = a.slice_range(&[1, 2], &[3, 6]).unwrap();
        assert_eq!(b.shape.dims, vec![2, 4]);
    });
}

#[test]
fn narrow_axis() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[10, 8]));
        let b = a.narrow(0, 2, 3).unwrap();
        assert_eq!(b.shape.dims, vec![3, 8]);
    });
}

// ── Comparison ops ──────────────────────────────────────────────────

#[test]
fn compare_produces_bool_dtype() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = cx.input("b", f32_shape(&[3, 4]));
        let c = a.lt(&b).unwrap();
        assert_eq!(c.shape.dtype, DType::Bool);
        assert_eq!(c.shape.dims, vec![3, 4]);
    });
}

#[test]
fn compare_with_broadcast() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = cx.input("b", f32_shape(&[4]));
        let c = a.ge(&b).unwrap();
        assert_eq!(c.shape.dtype, DType::Bool);
        assert_eq!(c.shape.dims, vec![3, 4]);
    });
}

// ── Select ──────────────────────────────────────────────────────────

#[test]
fn select_shape() {
    trace_with(|cx| {
        let pred = cx.input("p", Shape::new(vec![3, 4], DType::Bool));
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = cx.input("b", f32_shape(&[3, 4]));
        let c = Tensor::select(&pred, &a, &b).unwrap();
        assert_eq!(c.shape.dims, vec![3, 4]);
        assert_eq!(c.shape.dtype, DType::F32);
    });
}

// ── Scalar ops ──────────────────────────────────────────────────────

#[test]
fn scalar_mul_preserves_shape() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = (&a * 2.0).unwrap();
        assert_eq!(b.shape.dims, vec![3, 4]);
    });
}

#[test]
fn scalar_add_preserves_shape() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = (&a + 1.0).unwrap();
        assert_eq!(b.shape.dims, vec![3, 4]);
    });
}

#[test]
fn scalar_reverse_mul() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = (2.0 * &a).unwrap();
        assert_eq!(b.shape.dims, vec![3, 4]);
    });
}

// ── Gather (embedding lookup) ───────────────────────────────────────

#[test]
fn embedding_lookup_shape() {
    trace_with(|cx| {
        let table = cx.input("table", f32_shape(&[100, 32]));
        let indices = cx.input("idx", Shape::new(vec![8], DType::I32));
        let out = table.gather(&indices).unwrap();
        assert_eq!(out.shape.dims, vec![8, 32]);
        assert_eq!(out.shape.dtype, DType::F32);
    });
}

#[test]
fn embedding_lookup_batched_indices() {
    trace_with(|cx| {
        let table = cx.input("table", f32_shape(&[100, 32]));
        let indices = cx.input("idx", Shape::new(vec![2, 8], DType::I32));
        let out = table.gather(&indices).unwrap();
        assert_eq!(out.shape.dims, vec![2, 8, 32]);
    });
}

// ── Iota ────────────────────────────────────────────────────────────

#[test]
fn iota_shape() {
    let mut cx = TraceCx::new("test");
    let a = cx.iota(Shape::new(vec![4, 4], DType::I32), 0).unwrap();
    assert_eq!(a.shape.dims, vec![4, 4]);
    assert_eq!(a.shape.dtype, DType::I32);
}

// ── Softmax ─────────────────────────────────────────────────────────

#[test]
fn softmax_preserves_shape() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[2, 3, 4]));
        let b = a.softmax(-1).unwrap();
        assert_eq!(b.shape.dims, vec![2, 3, 4]);
    });
}

// ── Argmax ──────────────────────────────────────────────────────────

#[test]
fn argmax_last_axis() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[2, 3, 4]));
        let b = a.argmax(-1).unwrap();
        assert_eq!(b.shape.dims, vec![2, 3]);
        assert_eq!(b.shape.dtype, DType::I32);
    });
}

#[test]
fn argmax_first_axis() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[5, 10]));
        let b = a.argmax(0).unwrap();
        assert_eq!(b.shape.dims, vec![10]);
        assert_eq!(b.shape.dtype, DType::I32);
    });
}

#[test]
fn argmax_1d() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[8]));
        let b = a.argmax(0).unwrap();
        assert_eq!(b.shape.dims, Vec::<i64>::new());
        assert_eq!(b.shape.dtype, DType::I32);
    });
}

// ── Cos / Sin ───────────────────────────────────────────────────────

#[test]
fn cos_sin_preserve_shape() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[4, 8]));
        assert_eq!(a.cos().shape.dims, vec![4, 8]);
        assert_eq!(a.sin().shape.dims, vec![4, 8]);
    });
}

// ── Convert ─────────────────────────────────────────────────────────

#[test]
fn convert_i32_to_f32() {
    let mut cx = TraceCx::new("test");
    let a = cx.iota(Shape::new(vec![4], DType::I32), 0).unwrap();
    let b = a.to_dtype(DType::F32);
    assert_eq!(b.shape.dims, vec![4]);
    assert_eq!(b.shape.dtype, DType::F32);
}

#[test]
fn convert_noop_same_dtype() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = a.to_dtype(DType::F32);
        assert_eq!(b.value, a.value);
    });
}

// ── Expand ──────────────────────────────────────────────────────────

#[test]
fn expand_size_1_dim() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[2, 1, 4]));
        let b = a.expand(&[2, 3, 4]).unwrap();
        assert_eq!(b.shape.dims, vec![2, 3, 4]);
    });
}

#[test]
fn expand_noop_same_shape() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[2, 3]));
        let b = a.expand(&[2, 3]).unwrap();
        assert_eq!(b.value, a.value);
    });
}

#[test]
fn expand_non_one_dim_fails() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[2, 3]));
        assert!(a.expand(&[2, 5]).is_err());
    });
}
