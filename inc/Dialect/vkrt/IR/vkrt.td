
#ifndef VKRT_DIALECT_TD
#define VKRT_DIALECT_TD

include "mlir/IR/OpBase.td"

def vkrtDialect : Dialect {
    let name = "vkrt";
    let cppNamespace = "::vkrt";
    let summary = "vkrt dialect for vkml backend operations";
    let useDefaultTypePrinterParser = true;
    let dependentDialects = [
        "gpu::GPUDialect", "memref::MemRefDialect", "spirv::SPIRVDialect"
    ];
}

#endif
