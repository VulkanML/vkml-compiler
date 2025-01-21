#ifndef RT_DIALECT_TD
#define RT_DIALECT_TD

include "mlir/IR/OpBase.td"

def VKML_RT_Dialect : Dialect {
	let name = "rt";
	let cppNamespace = "vkml::runtime";
	let summary = "VKML dialect for vkml backend operations";
	let useDefaultTypePrinterParser = true;
	let dependentDialects = [
		"gpu::GPUDialect", "spirv::SPIRVDialect"
	];
}


#endif


#ifndef RT_OPS_TD
#define RT_OPS_TD

def rtOp<string mnemonic, list<Trait> traits = []> :
  Op<VKML_RT_Dialect, mnemonic, traits>;	

def rtCopy : rtOp<"copy"> ;


#endif