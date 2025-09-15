#pragma once
#include <vector>
#include "Tensor.h"

enum class OperationType { ADD, SUB, MUL, DIV, NEG, TOSA_OP, UNKNOWN };

class TensorExpression {
public:
    TensorExpression(const std::vector<Tensor>& inputs, OperationType opType, const Tensor& output);
    const std::vector<Tensor>& getInputs() const;
    OperationType getOperationType() const;
    const Tensor& getOutput() const;
private:
    std::vector<Tensor> inputs_;
    OperationType opType_;
    Tensor output_;
};
