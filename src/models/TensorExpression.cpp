#include "TensorExpression.h"

TensorExpression::TensorExpression(const std::vector<Tensor>& inputs, OperationType opType, const Tensor& output)
    : inputs_(inputs), opType_(opType), output_(output) {}

const std::vector<Tensor>& TensorExpression::getInputs() const { return inputs_; }
OperationType TensorExpression::getOperationType() const { return opType_; }
const Tensor& TensorExpression::getOutput() const { return output_; }
