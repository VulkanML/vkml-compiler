
#ifndef FRONTEND_DIALECT_COMPILER_H
#define FRONTEND_DIALECT_COMPILER_H
#include <mlir/IR/Types.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/Pass/PassManager.h>

using namespace mlir;
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Math/IR/Math.h>


#include "Dialect/frontend/frontendDialect.h.inc"
#define GET_OP_CLASSES
#include "Dialect/frontend/frontend.h.inc"
#define GET_TYPEDEF_CLASSES
#include "Dialect/frontend/frontendTypes.h.inc"

namespace {

}


#endif