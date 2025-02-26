#include "frontend.h"
#include <mlir/Pass/Pass.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/TypeID.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <llvm/ADT/TypeSwitch.h>

#include "Dialect/frontend/frontendDialect.cpp.inc"
#include "Dialect/frontend/frontendEnums.cpp.inc"
#include "Dialect/frontend/frontendTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/frontend/frontendAttr.cpp.inc"
	
#define GET_OP_CLASSES
#include "Dialect/frontend/frontend.cpp.inc"


void frontend::frontendDialect::initialize() {
	this->addOperations<
#define GET_OP_LIST
#include "Dialect/frontend/frontend.cpp.inc"
	>();
	this->addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/frontend/frontendAttr.cpp.inc"
	>();

}
