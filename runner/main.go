// cmd/run/main.go
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/gomlx/go-xla/pkg/installer"
	"github.com/gomlx/go-xla/pkg/pjrt"
)

func main() {
	mlirPath := flag.String("mlir", "forward.mlir", "Path to StableHLO MLIR file")
	pluginName := flag.String("plugin", "cpu", `PJRT plugin name (e.g. "cpu", "cuda") or full path to the plugin library`)
	flag.Parse()

	// This avoids Bazel: it downloads prebuilt PJRT plugins into a user-local lib dir if needed.
	// On macOS, installPath="" defaults under $HOME/Library/Application Support/; on Linux under $HOME/.local/lib/.
	if err := installer.AutoInstall("", true, installer.Normal); err != nil {
		log.Fatalf("installer.AutoInstall: %v", err)
	}

	mlirBytes, err := os.ReadFile(*mlirPath)
	if err != nil {
		log.Fatalf("read %s: %v", *mlirPath, err)
	}

	plugin, err := pjrt.GetPlugin(*pluginName)
	if err != nil {
		log.Fatalf("pjrt.GetPlugin(%q): %v", *pluginName, err)
	}

	client, err := plugin.NewClient(nil)
	if err != nil {
		log.Fatalf("plugin.NewClient: %v", err)
	}
	defer func() { _ = client.Destroy() }()

	exec, err := client.Compile().WithStableHLO(mlirBytes).Done()
	if err != nil {
		log.Fatalf("compile StableHLO: %v", err)
	}
	defer func() { _ = exec.Destroy() }()

	// Example inputs for your linear: (2x4) dot (4x3) + (3) bias.
	// Make sure your MLIR signature matches these shapes/dtypes.
	x := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}
	w := []float32{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		1, 1, 1,
	}
	b := []float32{10, 20, 30}

	bx, err := pjrt.ArrayToBuffer(client, x, 2, 4)
	if err != nil {
		log.Fatalf("ArrayToBuffer(x): %v", err)
	}
	defer func() { _ = bx.Destroy() }()

	bw, err := pjrt.ArrayToBuffer(client, w, 4, 3)
	if err != nil {
		log.Fatalf("ArrayToBuffer(w): %v", err)
	}
	defer func() { _ = bw.Destroy() }()

	bb, err := pjrt.ArrayToBuffer(client, b, 3)
	if err != nil {
		log.Fatalf("ArrayToBuffer(b): %v", err)
	}
	defer func() { _ = bb.Destroy() }()

	outs, err := exec.Execute(bx, bw, bb).Done()
	if err != nil {
		log.Fatalf("execute: %v", err)
	}
	defer func() {
		for _, o := range outs {
			_ = o.Destroy()
		}
	}()

	outFlat, outDims, err := pjrt.BufferToArray[float32](outs[0])
	if err != nil {
		log.Fatalf("BufferToArray(out): %v", err)
	}

	fmt.Printf("output dims=%v\n", outDims)
	fmt.Printf("output flat=%v\n", outFlat)
}
