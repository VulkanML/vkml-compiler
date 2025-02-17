
#ifndef VKRT_DIALECT_TD
#define VKRT_DIALECT_TD

include "mlir/IR/OpBase.td"

def backendDialect : Dialect {
    let name = "backend";
    let cppNamespace = "::backend";
    let summary = "vkrt dialect for vkml backend operations";
    let useDefaultTypePrinterParser = true;
    let dependentDialects = [
        "gpu::GPUDialect", "memref::MemRefDialect", "spirv::SPIRVDialect"
    ];
}

#endif
