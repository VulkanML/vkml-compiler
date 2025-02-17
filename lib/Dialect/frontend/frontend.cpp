#include "frontend.h"

using namespace mlir;

#include "Dialect/frontend/frontendDialect.cpp.inc"
#include "Dialect/frontend/frontendTypes.cpp.inc"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Interfaces/FunctionImplementation.h>



namespace frontend {
	void frontendDialect::initialize() {
		addOperations<
#define GET_OP_LIST
#include "Dialect/frontend/frontend.cpp.inc"
		>();

		addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/frontend/frontendTypes.cpp.inc"
		>();
	}

}


#define GET_OP_CLASSES
#include "Dialect/frontend/frontend.cpp.inc"