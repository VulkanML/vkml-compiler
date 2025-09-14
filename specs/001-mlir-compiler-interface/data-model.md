# Data Model: MLIR Compiler Interface Abstraction

## Entities

### Tensor
- Fields: shape (vector<int>), element_type (enum), symbolic_id (string)
- Relationships: Used as input/output for TensorExpression and OperationFunction

### TensorExpression
- Fields: inputs (list<Tensor>), operation_type (enum), output (Tensor)
- Relationships: References input tensors and operation function

### OperationFunction
- Fields: name (string), op_type (enum), input_types (list<type>), output_type (type)
- Relationships: Encapsulates a single tensor-level operation, invoked from main or segment

### ProgramSegment
- Fields: segment_id (string), ops (list<OperationFunction>), device_id (int)
- Relationships: Groups operation functions for scheduling/device assignment

### CompilationConfiguration
- Fields: target_devices (list<int>), debug_flags (list<string>), optimization_level (int)
- Relationships: Used during compilation

### DiagnosticReport
- Fields: errors (list<string>), warnings (list<string>), source_map (dict)
- Relationships: Produced during validation/compilation
