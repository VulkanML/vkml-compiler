#pragma once
#include <vector>
#include <string>
#include "Tensor.h"

enum class OpType { ADD, SUB, MUL, DIV, NEG, TOSA_OP, UNKNOWN };

class OperationFunction {
public:
    OperationFunction(const std::string& name, OpType opType, const std::vector<ElementType>& inputTypes, ElementType outputType);
    std::string getName() const;
    OpType getOpType() const;
    const std::vector<ElementType>& getInputTypes() const;
    ElementType getOutputType() const;
private:
    std::string name_;
    OpType opType_;
    std::vector<ElementType> inputTypes_;
    ElementType outputType_;
};
