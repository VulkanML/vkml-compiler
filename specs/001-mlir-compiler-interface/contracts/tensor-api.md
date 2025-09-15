# Tensor API Contract

## Overview
Defines the C++ API for tensor creation, operator overloading, and TOSA operations.

## Types
- `Tensor`: Represents a multi-dimensional array with shape and element type.

## Constructors
- `Tensor(const std::vector<int>& shape, ElementType type);`

## Operators
- `Tensor operator+(const Tensor& lhs, const Tensor& rhs);`
- `Tensor operator-(const Tensor& lhs, const Tensor& rhs);`
- `Tensor operator*(const Tensor& lhs, const Tensor& rhs);`
- `Tensor operator/(const Tensor& lhs, const Tensor& rhs);`
- `Tensor operator-() const;`

## Common Ops (TOSA)
- `Tensor abs(const Tensor& t);`
- `Tensor add(const Tensor& lhs, const Tensor& rhs);`
- `Tensor arithmetic_right_shift(const Tensor& lhs, const Tensor& rhs);`
- `Tensor avg_pool2d(const Tensor& input, const std::vector<int>& kernel, const std::vector<int>& stride, const std::vector<int>& pad);`
- `Tensor bitwise_and(const Tensor& lhs, const Tensor& rhs);`
- `Tensor bitwise_not(const Tensor& t);`
- `Tensor bitwise_or(const Tensor& lhs, const Tensor& rhs);`
- `Tensor bitwise_xor(const Tensor& lhs, const Tensor& rhs);`
- `Tensor cast(const Tensor& t, ElementType new_type);`
- `Tensor ceil(const Tensor& t);`
- `Tensor clamp(const Tensor& t, float min, float max);`
- `Tensor concat(const std::vector<Tensor>& tensors, int axis);`
- `Tensor conv2d(const Tensor& input, const Tensor& filter, const std::vector<int>& stride, const std::vector<int>& pad);`
- `Tensor depthwise_conv2d(const Tensor& input, const Tensor& filter, const std::vector<int>& stride, const std::vector<int>& pad);`
- `Tensor div(const Tensor& lhs, const Tensor& rhs);`
- `Tensor equal(const Tensor& lhs, const Tensor& rhs);`
- `Tensor exp(const Tensor& t);`
- `Tensor floor(const Tensor& t);`
- `Tensor fully_connected(const Tensor& input, const Tensor& weights, const Tensor& bias);`
- `Tensor gather(const Tensor& t, const Tensor& indices, int axis);`
- `Tensor global_avg_pool(const Tensor& input);`
- `Tensor global_max_pool(const Tensor& input);`
- `Tensor greater(const Tensor& lhs, const Tensor& rhs);`
- `Tensor greater_equal(const Tensor& lhs, const Tensor& rhs);`
- `Tensor hard_sigmoid(const Tensor& t);`
- `Tensor hard_swish(const Tensor& t);`
- `Tensor hard_tanh(const Tensor& t);`
- `Tensor identity(const Tensor& t);`
- `Tensor leaky_relu(const Tensor& t, float alpha);`
- `Tensor less(const Tensor& lhs, const Tensor& rhs);`
- `Tensor less_equal(const Tensor& lhs, const Tensor& rhs);`
- `Tensor log(const Tensor& t);`
- `Tensor logical_and(const Tensor& lhs, const Tensor& rhs);`
- `Tensor logical_not(const Tensor& t);`
- `Tensor logical_or(const Tensor& lhs, const Tensor& rhs);`
- `Tensor matmul(const Tensor& lhs, const Tensor& rhs);`
- `Tensor maximum(const Tensor& lhs, const Tensor& rhs);`
- `Tensor minimum(const Tensor& lhs, const Tensor& rhs);`
- `Tensor mul(const Tensor& lhs, const Tensor& rhs);`
- `Tensor neg(const Tensor& t);`
- `Tensor not_equal(const Tensor& lhs, const Tensor& rhs);`
- `Tensor one_hot(const Tensor& indices, int depth, float on_value, float off_value);`
- `Tensor pad(const Tensor& t, const std::vector<int>& paddings, float pad_value);`
- `Tensor pow(const Tensor& lhs, const Tensor& rhs);`
- `Tensor reduce_max(const Tensor& t, int axis);`
- `Tensor reduce_mean(const Tensor& t, int axis);`
- `Tensor reduce_min(const Tensor& t, int axis);`
- `Tensor reduce_prod(const Tensor& t, int axis);`
- `Tensor reduce_sum(const Tensor& t, int axis);`
- `Tensor relu(const Tensor& t);`
- `Tensor relu_n(const Tensor& t, float n);`
- `Tensor reshape(const Tensor& t, const std::vector<int>& new_shape);`
- `Tensor reverse(const Tensor& t, int axis);`
- `Tensor rsqrt(const Tensor& t);`
- `Tensor select(const Tensor& cond, const Tensor& x, const Tensor& y);`
- `Tensor sigmoid(const Tensor& t);`
- `Tensor sign(const Tensor& t);`
- `Tensor slice(const Tensor& t, const std::vector<int>& start, const std::vector<int>& size);`
- `Tensor softmax(const Tensor& t, int axis);`
- `Tensor split(const Tensor& t, int axis, int num_splits);`
- `Tensor sqrt(const Tensor& t);`
- `Tensor sub(const Tensor& lhs, const Tensor& rhs);`
- `Tensor table(const Tensor& t, const Tensor& table);`
- `Tensor tanh(const Tensor& t);`
- `Tensor tile(const Tensor& t, const std::vector<int>& multiples);`
- `Tensor transpose(const Tensor& t, const std::vector<int>& perm);`

## Segment Scoping
- `void begin_segment(const std::string& name, int device_id);`
- `void end_segment();`

## Compilation
- `void compile();`
- `void dump_ir(const std::string& stage);`
- `DiagnosticReport get_diagnostics();`

## Error Handling
- Throws `ShapeMismatchError` on invalid operations.
- Returns diagnostics for compilation errors.
