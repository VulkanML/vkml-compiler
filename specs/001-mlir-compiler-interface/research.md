# Phase 0 Research: MLIR Compiler Interface Abstraction

## Unknowns and Decisions

### 1. Tensor Operation Coverage
- Decision: Cover all core arithmetic ops (add, sub, mul, div, neg) and common tensor ops (matmul, reshape, broadcast) using TOSA, linalg, and tensor dialects.
- Rationale: These ops are most common in ML workloads and are well-supported in MLIR.
- Alternatives: Could restrict to TOSA only, but linalg/tensor needed for full coverage.

### 2. Segmentation Mechanism
- Decision: Segments are defined by scoped operations (begin_segment()/end_segment() or segment(name)), grouping contiguous tensor ops for scheduling and device assignment.
- Rationale: Explicit scoping is simple and user-friendly for v1; aligns with ml_program dialect.
- Alternatives: Automatic segmentation, but deferred to future versions.

### 3. Device Assignment
- Decision: Manual per-segment device assignment via device(id) modifier; unassigned segments run on device 0.
- Rationale: Simple, predictable, and matches user expectations for v1.
- Alternatives: Automatic load balancing, deferred to v2.

### 4. Dynamic Shapes
- Decision: Compiler will support dynamic shapes using MLIR bounds and shape inference.
- Rationale: MLIR supports dynamic shapes natively; user does not need to manage bounds.
- Alternatives: Restrict to static shapes, but would limit usability.

### 5. GPU Resource Exhaustion
- Decision: System will batch work and decompose large ops into loops over data chunks if GPU memory is insufficient.
- Rationale: Ensures robustness and usability for large workloads.
- Alternatives: Fail with error, but batching is more user-friendly.

### 6. Mixed Precision and Dtype Promotion
- Decision: Rely on MLIR's type system for promotion and validation.
- Rationale: MLIR provides strong type checking and promotion.
- Alternatives: Manual promotion, but unnecessary for v1.

### 7. Compilation Latency
- Decision: Compile subgraphs, not entire graphs, to keep latency low.
- Rationale: Users can control granularity via segmentation.
- Alternatives: Compile whole graph, but not needed for v1.

### 8. API Extension and Cancellation
- Decision: No extension API or cancellation support in v1.
- Rationale: Focus on core compiler functionality first.
- Alternatives: Add extension/cancellation, but defer to future versions.

## Best Practices
- Use MLIR's dialect registration and pass pipelines for modularity.
- Validate tensor shapes and types at expression formation.
- Provide clear diagnostics for errors.
- Document limitations and supported ops.

## Integration Patterns
- Use C++ operator overloading for frontend API.
- Use MLIR module/function structure for backend IR.
- Use ml_program dialect for multi-GPU orchestration.

## Summary
All major unknowns and clarifications from the spec have been resolved. The project is ready to proceed to Phase 1: Design & Contracts.
