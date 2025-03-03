#ifndef FRONTEND_DIALECT_COMPILER_H
#define FRONTEND_DIALECT_COMPILER_H
#include <mlir/IR/Types.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/BuiltinDialect.h>
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <mlir/Pass/PassManager.h>

using namespace mlir;
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Math/IR/Math.h>

#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>

#include "Dialect/frontend/frontendDialect.h.inc"
#include "Dialect/frontend/frontendEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "Dialect/frontend/frontendAttr.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/frontend/frontendTypes.h.inc"

#define GET_OP_CLASSES
#include "Dialect/frontend/frontend.h.inc"

#include "frontend_passes.h"

#endif