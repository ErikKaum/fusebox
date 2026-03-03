default:
    @just --list

build:
    cargo build

test:
    STABLEHLO_OPT=/Users/erikkaum/Documents/testing/stablehlo/bazel-bin/stablehlo-opt cargo test

clippy:
    cargo clippy -- -D warnings

fmt:
    cargo fmt

fmt-check:
    cargo fmt -- --check

ci: fmt-check clippy test

download-smollm2:
    uv run examples/smollm2/download-smollm2.py

download-pjrt:
    curl -L https://github.com/zml/pjrt-artifacts/releases/download/v0.2.2/pjrt-cpu_darwin-arm64.tar.gz -o pjrt-cpu.tar.gz
    tar -xzf pjrt-cpu.tar.gz
    rm pjrt-cpu.tar.gz

clean:
    cargo clean
    rm -f ./forward.mlir
