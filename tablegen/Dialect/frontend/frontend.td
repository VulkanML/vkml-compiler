#ifndef VKML_DIALECT_TD
#define VKML_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"


def frontendDialect : Dialect {
    let name = "frontend";
    let cppNamespace = "::frontend";
    let summary = "VKML dialect for vkml frontend operations";

    let dependentDialects = [
        "mlir::tensor::TensorDialect", 
        "mlir::affine::AffineDialect", 
        "mlir::arith::ArithDialect",
        "mlir::math::MathDialect",
        "mlir::func::FuncDialect"
    ];
}

class frontendOp<string mnemonic, list<Trait> traits = []> :
  Op<frontendDialect, mnemonic, traits>;

def frontend_abs : frontendOp<"abs">
{
    let summary = "Absolute value operation";
    let description = [{
        "This operation computes the absolute value of the input tensor element-wise."
    }];
    let arguments = (ins AnyTensor:$source);
    let results = (outs AnyTensor:$dest);
    let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";
}

#endif
