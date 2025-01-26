#ifndef VKML_COMPILER_H
#define VKML_COMPILER_H

#include "vkmlDialect.h"

#include <mlir/IR/Types.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>


namespace vkml_compiler {


class compiler {
	mlir::MLIRContext ctx;
	mlir::ModuleOp global_module;
	mlir::OpBuilder global_builder;
	std::vector<mlir::ModuleOp> device_modules;
public:
	compiler();
	~compiler();
	
	void addDevice(uint32_t device_id, uint32_t vendor_id, uint32_t device_type_id, const std::vector<std::string>& capabilities, const std::vector<std::string>& extensions);

	void initialize();
};

}

#endif