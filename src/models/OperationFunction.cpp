#include "OperationFunction.h"

OperationFunction::OperationFunction(const std::string& name, OpType opType, const std::vector<ElementType>& inputTypes, ElementType outputType)
    : name_(name), opType_(opType), inputTypes_(inputTypes), outputType_(outputType) {}

std::string OperationFunction::getName() const { return name_; }
OpType OperationFunction::getOpType() const { return opType_; }
const std::vector<ElementType>& OperationFunction::getInputTypes() const { return inputTypes_; }
ElementType OperationFunction::getOutputType() const { return outputType_; }
