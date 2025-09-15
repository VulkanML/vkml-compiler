# Feature Specification: MLIR Compiler Interface Abstraction for Tensor & TOSA Operator Overloading

**Feature Branch**: `001-mlir-compiler-interface`  
**Created**: 2025-09-10  
**Status**: Draft  
**Input**: User description: "I want to build a mlir compiler interface that abstracts tensor and tosa op generation into a simplified interface so that I can use C++ overloaded operators to generate the backend GPU code. I am also hoping to use the ml_program dialect to handle multi GPU use cases."

## Execution Flow (main)
```
1. Parse user description from Input
   → If empty: ERROR "No feature description provided"
2. Extract key concepts from description
   → Identify: actors (C++ developer / library user), actions (define computations with operators), data (tensor shapes / operations), constraints (multi-GPU orchestration)
3. For each unclear aspect:
   → Mark with [NEEDS CLARIFICATION: specific question]
4. Fill User Scenarios & Testing section
   → If no clear user flow: ERROR "Cannot determine user scenarios"
5. Generate Functional Requirements
   → Each requirement must be testable
   → Mark ambiguous requirements
6. Identify Key Entities (if data involved)
7. Run Review Checklist
   → If any [NEEDS CLARIFICATION]: WARN "Spec has uncertainties"
   → If implementation details found: ERROR "Remove tech details"
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no specific pass pipelines, dialect registration code, or CMake details)
- 👥 Written for stakeholders assessing developer productivity and GPU capability enablement

### Section Requirements
- All mandatory sections included
- Optional sections removed unless materially relevant

### Ambiguity Handling Rules
- Mark unexplained performance targets, memory constraints, distribution strategies, failure semantics, and scheduling policies with [NEEDS CLARIFICATION]

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
A C++ developer wants to express tensor computations (e.g., elementwise add, matmul, reductions) using natural C++ operator overloading (e.g., `a + b * c`) and have the system automatically produce MLIR with TOSA (or tensor) dialect ops, progressively lower it toward GPU-executable form, and coordinate execution across multiple GPUs using high-level program constructs.

### Acceptance Scenarios
1. **Given** a developer includes the high-level interface and defines `Tensor C = A + B;`, **When** they trigger compilation, **Then** a generated MLIR module contains distinct functions for each high-level op and a main orchestration function.
2. **Given** a computation combining multiple chained ops, **When** the developer invokes a build step, **Then** the system emits MLIR that can be lowered through standard pipelines without manual dialect knowledge required by the developer.
3. **Given** a system with multiple GPUs available, **When** the developer annotates or partitions a computation, **Then** the specification allows orchestrating logical program segments across devices via an ml_program-like abstraction. The initial version will have a manual per-segment device assignment via a device(id) modifier; unassigned segments run on device 0. No automatic load balancing in v1.
4. **Given** an invalid tensor shape mismatch in an overloaded operator expression, **When** compilation is attempted, **Then** the user is provided a clear diagnostic referencing the source expression.

### Edge Cases
- What happens when tensors have dynamic shapes? The mlir-compiler should be able to handle this via bounds and will deal with the dynamic shapes. 
- How does the interface behave when GPU resources are exhausted? The system should look at batching work to run on the gpu given the memory constraint. If there is no more memory for the operation we decompose the operation into a loop to loop over chunks of the data to be computed by the GPU. 
- What is the expected behavior if only a single GPU is present? (Fallback vs. error) this is not a concern as the devices will be manually specified for v1
- How are mixed precision or dtype promotions handled? This should be handled by the mlir compiler as there is a strong type checking and promotion system builtin
- How are large graphs with thousands of ops managed for compilation latency? compilation latency is not a concern as we want to run subgraphs of the graph not the larger graph. 

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: System MUST allow constructing tensor objects via a simple C++ API representing shape and element type.
- **FR-002**: System MUST support operator overloading for core tensor arithmetic: addition, subtraction, multiplication, division, unary negation.
- **FR-003**: System MUST support common tensor operations beyond arithmetic (e.g., matmul, reshape, broadcast) the linalg and tensor dialects need to be covered for the system to work.
- **FR-004**: System MUST generate an MLIR module where each high-level operation is wrapped in an individual function and invoked from a main coordination function.
- **FR-005**: System MUST ensure generated IR uses TOSA (or fallback tensor/linalg) ops at the high level before lowering.
- **FR-006**: System MUST provide a build/compile entrypoint that returns success/failure and diagnostics.
- **FR-007**: System MUST validate operand shape compatibility at expression formation time.
- **FR-008**: System MUST produce human-readable error messages when an expression is invalid.
- **FR-009**: System MUST allow users to request MLIR textual output of intermediate stages (high-level, post-bufferization, pre-GPU) high-level and per-gpu.
- **FR-010**: System MUST support configuration of target GPU backend(s) without requiring user knowledge of MLIR dialect internals. For now just use the GPU
- **FR-011**: System MUST provide a mechanism to express device placement for multi-GPU scenarios via manual annotation (device(id) modifier); unassigned segments run on device 0.
- **FR-012**: System MUST allow grouping a sequence of tensor ops into a higher-level program segment aligned with ml_program dialect semantics. A segment is defined by a scoped operation (e.g., begin_segment()/end_segment() or segment(name)), grouping contiguous tensor ops for scheduling and device assignment. Only manual grouping is supported in v1.
- **FR-013**: System MUST allow retrieval of compiled artifact metadata (e.g., op count, device usage) for introspection.
- **FR-014**: System MUST support deterministic builds given identical input expressions and configuration.
- **FR-015**: System MUST expose a stable API for extension (adding custom ops) Not necessary for v1.
- **FR-016**: System MUST enable users to cancel or abort a long-running compilation not needed for v1 we just want the compiler not the execution engine.
- **FR-017**: System MUST document limitations (unsupported ops, shape constraints) in a discoverable way.
- **FR-018**: System MUST isolate multi-GPU execution concerns from single-GPU users (no extra required complexity if only one device).

### Key Entities
- **Tensor Expression**: Represents a lazily constructed computation graph node with references to input tensors and an operation type (no execution detail).
- **Tensor**: Abstract data handle containing shape metadata, element type, and symbolic identity (not raw memory description at spec level).
- **Operation Function**: A generated MLIR function encapsulating a single tensor-level operation.
- **Program Segment**: A logical grouping of operation functions intended for potential multi-device scheduling.
- **Compilation Configuration**: User-provided or default settings (target devices, debug output flags, optimization toggles). 
- **Diagnostic Report**: Structured collection of errors/warnings produced during expression validation or compilation.

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [ ] No implementation details (languages, frameworks, APIs) — Verified
- [ ] Focused on user value and business needs — Verified
- [ ] Written for non-technical stakeholders — Verified
- [ ] All mandatory sections completed — Verified

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain — Pending
- [ ] Requirements are testable and unambiguous — Pending (several clarifications needed)
- [ ] Success criteria are measurable — Partially (missing performance metrics)
- [ ] Scope is clearly bounded — Partially (multi-GPU behaviors unclear)
- [ ] Dependencies and assumptions identified — Partial (needs device & backend assumptions)

---

## Execution Status
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [ ] Review checklist passed (awaiting clarifications)

---
