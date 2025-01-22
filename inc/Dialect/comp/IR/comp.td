#ifndef COMP_DIALECT_TD
#define COMP_DIALECT_TD

include "mlir/IR/OpBase.td"

def compDialect : Dialect {
    let name = "comp";
    let cppNamespace = "::comp";
    let summary = "compute dialect for vkml middleware operations";
    let useDefaultTypePrinterParser = true;
    let dependentDialects = [
        "tensor::TensorDialect", "gpu::GPUDialect", "affine::AffineDialect", "arith::ArithDialect"		
    ];
}

#endif
