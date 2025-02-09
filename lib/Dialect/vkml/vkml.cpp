#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#define GET_OP_CLASSES 
// #include "Dialect/vkml/IR/vkml.cpp.inc"
// #include "Dialect/vkml/IR/vkmlDialect.cpp.inc"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#define GET_TYPEDEF_CLASSES
// #include "Dialect/vkml/IR/vkmlTypes.cpp.inc"



// void vkmlDialect::initialize() {
// 	addOperations<
// #define GET_OP_LIST
// #include "Dialect/vkml/IR/vkml.cpp.inc" >();
// 		addTypes<
// #define GET_TYPEDEF_LIST
// #include "Dialect/vkml/IR/vkmlTypes.cpp.inc"
// 		>();
// }

