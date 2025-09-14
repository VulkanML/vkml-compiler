# Compiler API Contract

## Overview
Defines the interface for compiling tensor expressions, segment management, and diagnostics.

## Methods
- `void compile();` — Compiles all defined tensor expressions and segments into MLIR IR.
- `void dump_ir(const std::string& stage);` — Dumps MLIR IR at the specified stage ("high-level", "post-bufferization", "pre-gpu").
- `DiagnosticReport get_diagnostics();` — Returns compilation diagnostics.

## Segment Management
- `void begin_segment(const std::string& name, int device_id);` — Starts a new segment for grouping ops and device assignment. Each segment is represented as an `ml_program.subgraph` operation in the MLIR IR, with attributes for name and device assignment.
- `void end_segment();` — Ends the current segment.

## Metadata
- `ArtifactMetadata get_metadata();` — Returns metadata (op count, device usage, etc.).

## Error Handling
- Compilation errors are reported via `DiagnosticReport`.
- Invalid segment or device assignment throws `SegmentError`.
