use fusebox::dtype::DType;
use fusebox::error::Error;
use fusebox::shape::Shape;
use fusebox::tensor::Tensor;
use fusebox::trace::TraceCx;

fn f32_shape(dims: &[i64]) -> Shape {
    Shape::new(dims.to_vec(), DType::F32)
}

fn trace_with<F, T>(f: F) -> T
where
    F: FnOnce(&mut TraceCx) -> T,
{
    let mut cx = TraceCx::new("test");
    f(&mut cx)
}

// ── DType mismatch ──────────────────────────────────────────────────

#[test]
fn add_dtype_mismatch() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[4]));
        let b = cx.input("b", Shape::new(vec![4], DType::I32));
        let err = a.add(&b).unwrap_err();
        assert!(matches!(err, Error::DTypeMismatch { .. }));
    });
}

// ── Broadcast failures ──────────────────────────────────────────────

#[test]
fn broadcast_incompatible_shapes() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = cx.input("b", f32_shape(&[5]));
        let err = a.add(&b).unwrap_err();
        assert!(matches!(err, Error::BroadcastError { .. }));
    });
}

#[test]
fn broadcast_incompatible_3d() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[2, 3, 4]));
        let b = cx.input("b", f32_shape(&[2, 5, 4]));
        let err = a.mul(&b).unwrap_err();
        assert!(matches!(err, Error::BroadcastError { .. }));
    });
}

// ── Matmul errors ───────────────────────────────────────────────────

#[test]
fn matmul_rank_1() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[4]));
        let b = cx.input("b", f32_shape(&[4, 5]));
        let err = a.matmul(&b).unwrap_err();
        assert!(matches!(err, Error::RankMismatch { .. }));
    });
}

#[test]
fn matmul_contracting_dim_mismatch() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = cx.input("b", f32_shape(&[5, 6]));
        let err = a.matmul(&b).unwrap_err();
        assert!(matches!(err, Error::DimMismatch { .. }));
    });
}

// ── Reshape errors ──────────────────────────────────────────────────

#[test]
fn reshape_element_count_mismatch() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let err = a.reshape(&[10]).unwrap_err();
        assert!(matches!(err, Error::InvalidParam { .. }));
    });
}

// ── Transpose errors ────────────────────────────────────────────────

#[test]
fn transpose_wrong_permutation_length() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let err = a.transpose(&[0, 1, 2]).unwrap_err();
        assert!(matches!(err, Error::InvalidParam { .. }));
    });
}

// ── Squeeze errors ──────────────────────────────────────────────────

#[test]
fn squeeze_non_one_dim() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let err = a.squeeze(0).unwrap_err();
        assert!(matches!(err, Error::InvalidParam { .. }));
    });
}

// ── Reduction axis out of range ─────────────────────────────────────

#[test]
fn reduce_axis_out_of_range() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[3, 4]));
        let err = a.sum(&[5]).unwrap_err();
        assert!(matches!(err, Error::InvalidParam { .. }));
    });
}

// ── Concatenate errors ──────────────────────────────────────────────

#[test]
fn cat_rank_mismatch() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[2, 4]));
        let b = cx.input("b", f32_shape(&[3, 4, 5]));
        let err = Tensor::cat(&[&a, &b], 0).unwrap_err();
        assert!(matches!(err, Error::InvalidParam { .. }));
    });
}

#[test]
fn cat_non_concat_dim_mismatch() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[2, 3]));
        let b = cx.input("b", f32_shape(&[4, 5]));
        let err = Tensor::cat(&[&a, &b], 0).unwrap_err();
        assert!(matches!(err, Error::InvalidParam { .. }));
    });
}

// ── Slice errors ────────────────────────────────────────────────────

#[test]
fn slice_out_of_bounds() {
    trace_with(|cx| {
        let a = cx.input("a", f32_shape(&[4, 8]));
        let err = a.slice_range(&[0, 0], &[5, 8]).unwrap_err();
        assert!(matches!(err, Error::InvalidParam { .. }));
    });
}

// ── Select errors ───────────────────────────────────────────────────

#[test]
fn select_non_bool_pred() {
    trace_with(|cx| {
        let pred = cx.input("p", f32_shape(&[3, 4]));
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = cx.input("b", f32_shape(&[3, 4]));
        let err = Tensor::select(&pred, &a, &b).unwrap_err();
        assert!(matches!(err, Error::InvalidParam { .. }));
    });
}

#[test]
fn select_shape_mismatch() {
    trace_with(|cx| {
        let pred = cx.input("p", Shape::new(vec![3, 4], DType::Bool));
        let a = cx.input("a", f32_shape(&[3, 4]));
        let b = cx.input("b", f32_shape(&[3, 5]));
        let err = Tensor::select(&pred, &a, &b).unwrap_err();
        assert!(matches!(err, Error::ShapeMismatch { .. }));
    });
}

// ── Embedding errors ────────────────────────────────────────────────

#[test]
fn gather_non_integer_indices() {
    trace_with(|cx| {
        let table = cx.input("table", f32_shape(&[100, 32]));
        let indices = cx.input("idx", f32_shape(&[8]));
        let err = table.gather(&indices).unwrap_err();
        assert!(matches!(err, Error::InvalidParam { .. }));
    });
}

#[test]
fn gather_non_2d_table() {
    trace_with(|cx| {
        let table = cx.input("table", f32_shape(&[100]));
        let indices = cx.input("idx", Shape::new(vec![8], DType::I32));
        let err = table.gather(&indices).unwrap_err();
        assert!(matches!(err, Error::InvalidParam { .. }));
    });
}

// ── Graph mismatch ──────────────────────────────────────────────────

#[test]
fn graph_mismatch_on_add() {
    let mut cx1 = TraceCx::new("graph1");
    let mut cx2 = TraceCx::new("graph2");
    let a = cx1.input("a", f32_shape(&[4]));
    let b = cx2.input("b", f32_shape(&[4]));
    let err = a.add(&b).unwrap_err();
    assert!(matches!(err, Error::GraphMismatch));
}

// ── Scoped error ────────────────────────────────────────────────────

#[test]
fn scoped_error_display() {
    let err = Error::InvalidParam {
        msg: "bad shape".into(),
    };
    let scoped = err.with_scope("layer/attn");
    let msg = format!("{}", scoped);
    assert!(msg.contains("layer/attn"));
    assert!(msg.contains("bad shape"));
}

// ── Validation error ────────────────────────────────────────────────

#[test]
fn validation_error_display() {
    let err = Error::ValidationError("weight mismatch".into());
    let msg = format!("{}", err);
    assert!(msg.contains("validation error"));
    assert!(msg.contains("weight mismatch"));
}
