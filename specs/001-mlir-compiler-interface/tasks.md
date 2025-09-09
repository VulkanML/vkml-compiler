# Tasks: MLIR Compiler Interface Abstraction

**Feature Branch**: 001-mlir-compiler-interface

## Parallel Execution Guidance
- Tasks marked [P] can be executed in parallel.
- Tasks in the same file or with dependencies must be executed sequentially.

---

## Numbered Task List

### Setup
T001. Initialize project structure, dependencies, and build system. (src/, tests/, contracts/, etc.)
T002. Set up MLIR, C++ toolchain, and test framework (e.g., CMake, GoogleTest).

### Contract/API Tests [P]
T003. Write contract tests for tensor API (`contracts/tensor-api.md`). [P]
T004. Write contract tests for compiler API (`contracts/compiler-api.md`). [P]

### Quickstart/Integration Tests [P]
T005. Write integration test for tensor creation and operator chaining (see quickstart.md). [P]
T006. Write integration test for segment assignment and device selection. [P]
T007. Write integration test for compilation diagnostics and error handling. [P]

### Core Model Implementation [P]
T008. Implement Tensor entity (data-model.md). [P]
T009. Implement TensorExpression entity. [P]
T010. Implement OperationFunction entity. [P]
T011. Implement ProgramSegment entity, mapping to ml_program.subgraph. [P]
T012. Implement CompilationConfiguration entity. [P]
T013. Implement DiagnosticReport entity. [P]

### Operator and TOSA Op Wrappers [P]
T014. Implement operator overloading for Tensor (+, -, *, /, negation). [P]
T015. Implement all TOSA op wrappers as defined in tensor-api.md. [P]

### Compiler Logic & Integration
T016. Implement compiler logic for IR generation, segment management, and device assignment.
T017. Implement MLIR pass pipeline orchestration.
T018. Integrate segment management with MLIR ml_program.subgraph ops.
T019. Integrate device assignment and metadata reporting.
T020. Implement error handling and diagnostics.

### Polish & Validation [P]
T021. Add unit tests for edge cases (shape mismatch, dynamic shapes, device assignment). [P]
T022. Add documentation and usage examples. [P]
T023. Optimize performance and validate deterministic builds. [P]

---

## Dependency Notes
- Setup tasks (T001-T002) must be completed first.
- Contract and integration tests (T003-T007) before implementation (TDD).
- Model/entity and operator tasks (T008-T015) can run in parallel after setup and tests.
- Compiler logic and integration (T016-T020) follow model/entity completion.
- Polish tasks (T021-T023) can run in parallel after core implementation.
