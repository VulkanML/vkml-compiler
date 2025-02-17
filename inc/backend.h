#ifndef BACKEND_DIALECT_COMPILER_H
#define BACKEND_DIALECT_COMPILER_H

#include <mlir/IR/Types.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/BuiltinDialect.h>


#include "Dialect/backend/backendDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "Dialect/backend/backendTypes.h.inc"

#define GET_OP_CLASSES
#include "Dialect/backend/backend.h.inc"



#endif