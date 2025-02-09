#ifndef VKML_DIALECT_TD
#define VKML_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def vkmlDialect : Dialect {
    let name = "vkml_dialect";
    let cppNamespace = "::comp";
    let summary = "VKML dialect for vkml frontend operations";
    let useDefaultTypePrinterParser = true;
    let dependentDialects = [
        "tensor::TensorDialect", "affine::AffineDialect", "arith::ArithDialect" , "spirv::SpirvDialect"
    ];
}

def vkmlTensor : AnyTypeOf<[AnyRankedTensor], "", "::mlir::ShapedType">;
class vkmlType<string name, list<Trait> traits=[]>: TypeDef<vkmlDialect, name, traits>;

class vkmlOp<string mnemonic, list<Trait> traits = []> :
  Op<vkmlDialect, mnemonic, traits>;

def vkml_abs : vkmlOp<"abs">
{
    let summary = "Absolute value operation";
    let description = [{
        "This operation computes the absolute value of the input tensor element-wise."
    }];
    let arguments = (ins AnyRankedTensor:$operand);
    let results = (outs AnyRankedTensor:$result);
    
    let builders = [
        OpBuilder<(ins "Value":$x)>
    ];
}

#endif
