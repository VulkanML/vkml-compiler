# Quickstart: MLIR Compiler Interface Abstraction

## Getting Started

1. Install the library and dependencies.
2. Include the C++ header for tensor and operator API.
3. Define tensors: `Tensor A(shape, type); Tensor B(shape, type);`
4. Express computation: `Tensor C = A + B;`
5. Optionally, group ops in a segment: `begin_segment("seg1", device_id); ... end_segment();`
6. Compile: `compiler.compile();`
7. Inspect MLIR output: `compiler.dump_ir(stage);`
8. Check diagnostics: `compiler.get_diagnostics();`

## Example
```cpp
Tensor A({2,3}, FLOAT32);
Tensor B({2,3}, FLOAT32);
begin_segment("seg1", 1); // Assign to GPU 1
Tensor C = A + B;
end_segment();
compiler.compile();
compiler.dump_ir("high-level");
```

## Test Scenario
- Given valid tensor shapes, compilation succeeds and MLIR IR is generated.
- Given shape mismatch, compilation fails with diagnostic.
- Given segment assignment, ops are grouped and device is set in IR.
