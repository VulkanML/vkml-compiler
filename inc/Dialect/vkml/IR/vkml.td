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
    let name = "vkml";
    let cppNamespace = "::vkml";
    let summary = "VKML dialect for vkml frontend operations";
    let useDefaultTypePrinterParser = true;
    let dependentDialects = [
        "tensor::TensorDialect", "affine::AffineDialect", "arith::ArithDialect"		
    ];
}

def vkmlTensor : AnyTypeOf<[AnyRankedTensor], "", "::mlir::ShapedType">;
class vkmlType<string name, list<Trait> traits=[]>: TypeDef<vkmlDialect, name, traits>;

class vkmlOp<string mnemonic, list<Trait> traits = []> :
  Op<vkmlDialect, mnemonic, traits>;

class vkmlUnaryOp<string mnemonic, list<Trait> traits = []> :
  vkmlOp<mnemonic, traits> {
    let arguments = (ins AnyRankedOrUnrankedMemRef:$operand);
    let results = (outs AnyRankedOrUnrankedMemRef:$result);
    let assemblyFormat = "$inputs attr-dict `:` type($result)";
}

class vkmlBinaryOp<string mnemonic, list<Trait> traits = []> :
  vkmlOp<mnemonic, traits> {
    let arguments = (ins AnyRankedOrUnrankedMemRef:$lhs, AnyRankedOrUnrankedMemRef:$rhs);
    let results = (outs AnyRankedOrUnrankedMemRef:$result);
    let assemblyFormat = "$inputs attr-dict `:` type($result)";
}

#endif
