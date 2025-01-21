#ifndef COMP_DIALECT_TD
#define COMP_DIALECT_TD

include "mlir/IR/OpBase.td"

def VKML_COMP_Dialect : Dialect {
	let name = "comp";
	let cppNamespace = "vkml::comp";
	let summary = "VKML dialect for vkml middleware operations";
	let useDefaultTypePrinterParser = true;
	let dependentDialects = [
		"gpu::GPUDialect", "affine::AffineDialect", "arith::ArithDialect"		
	];
}

#endif
