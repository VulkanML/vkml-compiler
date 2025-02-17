#ifndef VKML_DIALECT_TD
#define VKML_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def frontendDialect : Dialect {
    let name = "frontend";
    let cppNamespace = "::frontend";
    let summary = "VKML dialect for vkml frontend operations";
  
    let useDefaultAttributePrinterParser = 1;

    let dependentDialects = [
        "mlir::tensor::TensorDialect", 
        "mlir::func::FuncDialect"
    ];
}

class frontendOp<string mnemonic, list<Trait> traits = []> :
  Op<frontendDialect, mnemonic, traits>;

class frontend_unary_op<string mnemonic, list<Trait> traits = []> :
    frontendOp<mnemonic, traits> {
        let arguments = (ins AnyTensor:$src);
        let results = (outs AnyTensor:$dest);
        let assemblyFormat = "$src attr-dict `:` type($src) `to` type($dest)";
}

class frontend_binary_op<string mnemonic, list<Trait> traits = []> :
    frontendOp<mnemonic, traits> {
        let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
        let results = (outs AnyTensor:$dest);
        let assemblyFormat = "$lhs`,` $rhs attr-dict `:` type($lhs)`,` type($rhs) `to` type($dest)";
}


def frontend_abs : I32EnumAttrCase<"ABS", 0, "abs">;
def frontend_neg : I32EnumAttrCase<"NEG", 1, "neg">;
def frontend_sqrt : I32EnumAttrCase<"SQRT", 2, "sqrt">;
def frontend_exp : I32EnumAttrCase<"EXP", 3, "exp">;
def frontend_log : I32EnumAttrCase<"LOG", 4, "log">;
def frontend_sin : I32EnumAttrCase<"SIN", 5, "sin">;
def frontend_cos : I32EnumAttrCase<"COS", 6, "cos">;
def frontend_tan : I32EnumAttrCase<"TAN", 7, "tan">;
def frontend_asin : I32EnumAttrCase<"ASIN", 8, "asin">;
def frontend_acos : I32EnumAttrCase<"ACOS", 9, "acos">;
def frontend_atan : I32EnumAttrCase<"ATAN", 10, "atan">;
def frontend_sinh : I32EnumAttrCase<"SINH", 11, "sinh">;
def frontend_cosh : I32EnumAttrCase<"COSH", 12, "cosh">;
def frontend_tanh : I32EnumAttrCase<"TANH", 13, "tanh">;
def frontend_asinh : I32EnumAttrCase<"ASINH", 14, "asinh">;
def frontend_acosh : I32EnumAttrCase<"ACOSH", 15, "acosh">;
def frontend_atanh : I32EnumAttrCase<"ATANH", 16, "atanh">;
def frontend_ceil : I32EnumAttrCase<"CEIL", 17, "ceil">;
def frontend_floor : I32EnumAttrCase<"FLOOR", 18, "floor">;
def frontend_round : I32EnumAttrCase<"ROUND", 19, "round">;
def frontend_trunc : I32EnumAttrCase<"TRUNC", 20, "trunc">;

def frontend_add : I32EnumAttrCase<"ADD", 0, "add">;
def frontend_sub : I32EnumAttrCase<"SUB", 1, "sub">;
def frontend_mul : I32EnumAttrCase<"MUL", 2, "mul">;
def frontend_div : I32EnumAttrCase<"DIV", 3, "div">;
def frontend_mod : I32EnumAttrCase<"MOD", 4, "mod">;
def frontend_pow : I32EnumAttrCase<"POW", 5, "pow">;

def frontend_reduce_sum : I32EnumAttrCase<"REDUCE_SUM", 0, "reduce_sum">;
def frontend_reduce_max : I32EnumAttrCase<"REDUCE_MAX", 1, "reduce_max">;
def frontend_reduce_min : I32EnumAttrCase<"REDUCE_MIN", 2, "reduce_min">;
def frontend_reduce_mean : I32EnumAttrCase<"REDUCE_MEAN", 3, "reduce_mean">;
def frontend_reduce_prod : I32EnumAttrCase<"REDUCE_PROD", 4, "reduce_prod">;
def frontend_reduce_l0 : I32EnumAttrCase<"REDUCE_L0", 5, "reduce_l0">;
def frontend_reduce_l1 : I32EnumAttrCase<"REDUCE_L1", 6, "reduce_l1">;

def frontend_bitwise_and : I32EnumAttrCase<"BITWISE_AND", 0, "bitwise_and">;
def frontend_bitwise_or : I32EnumAttrCase<"BITWISE_OR", 1, "bitwise_or">;
def frontend_bitwise_xor : I32EnumAttrCase<"BITWISE_XOR", 2, "bitwise_xor">;
def frontend_bitwise_not : I32EnumAttrCase<"BITWISE_NOT", 3, "bitwise_not">;
def frontend_bitwise_left_shift : I32EnumAttrCase<"BITWISE_LEFT_SHIFT", 4, "bitwise_left_shift">;
def frontend_bitwise_right_shift : I32EnumAttrCase<"BITWISE_RIGHT_SHIFT", 5, "bitwise_right_shift">;

def frontend_logical_and : I32EnumAttrCase<"LOGICAL_AND", 0, "logical_and">;
def frontend_logical_or : I32EnumAttrCase<"LOGICAL_OR", 1, "logical_or">;
def frontend_logical_not : I32EnumAttrCase<"LOGICAL_NOT", 2, "logical_not">;

def frontend_relational_lt : I32EnumAttrCase<"RELATIONAL_LT", 0, "relational_lt">;
def frontend_relational_le : I32EnumAttrCase<"RELATIONAL_LE", 1, "relational_le">;
def frontend_relational_gt : I32EnumAttrCase<"RELATIONAL_GT", 2, "relational_gt">;
def frontend_relational_ge : I32EnumAttrCase<"RELATIONAL_GE", 3, "relational_ge">;
def frontend_relational_eq : I32EnumAttrCase<"RELATIONAL_EQ", 4, "relational_eq">;
def frontend_relational_ne : I32EnumAttrCase<"RELATIONAL_NE", 5, "relational_ne">;



def UnaryArithEnum : I32EnumAttr<"UnaryArithEnum", "an enum", [
                            frontend_abs,
                            frontend_neg,
                            frontend_sqrt,
                            frontend_exp,
                            frontend_log,
                            frontend_sin,
                            frontend_cos,
                            frontend_tan,
                            frontend_asin,
                            frontend_acos,
                            frontend_atan,
                            frontend_sinh,
                            frontend_cosh,
                            frontend_tanh,
                            frontend_asinh,
                            frontend_acosh,
                            frontend_atanh,
                            frontend_ceil,
                            frontend_floor,
                            frontend_round,
                            frontend_trunc   ]> {
  
  let cppNamespace = "::frontend";
  let genSpecializedAttr = 0;
}

def UnaryArithEnumAttr : EnumAttr<frontendDialect, UnaryArithEnum, "unary_arith_enum">{
      let valueType = I32;
      let cppNamespace = "::frontend";
      let assemblyFormat = "`<` $value `>`";
}

def BinaryArithEnum : I32EnumAttr<"BinaryArithEnum", "an enum", [
                            frontend_add,
                            frontend_sub,
                            frontend_mul,
                            frontend_div,
                            frontend_mod,
                            frontend_pow ]> {
  
  let cppNamespace = "::frontend";
  let genSpecializedAttr = 0;
}

def BinaryArithEnumAttr : EnumAttr<frontendDialect, BinaryArithEnum, "binary_arith_enum">{
      let valueType = I32;
      let cppNamespace = "::frontend";
      let assemblyFormat = "`<` $value `>`";
}

def ReductionEnum : I32EnumAttr<"ReductionEnum", "an enum", [
                            frontend_reduce_sum,
                            frontend_reduce_max,
                            frontend_reduce_min,
                            frontend_reduce_mean,
                            frontend_reduce_prod,
                            frontend_reduce_l0,
                            frontend_reduce_l1 ]> {
  
  let cppNamespace = "::frontend";
  let genSpecializedAttr = 0;
}

def ReductionEnumAttr : EnumAttr<frontendDialect, ReductionEnum, "reduction_enum">{
      let valueType = I32;
      let cppNamespace = "::frontend";
      let assemblyFormat = "`<` $value `>`";
}

def BitwiseEnum : I32EnumAttr<"BitwiseEnum", "an enum", [
                            frontend_bitwise_and,
                            frontend_bitwise_or,
                            frontend_bitwise_xor,
                            frontend_bitwise_not,
                            frontend_bitwise_left_shift,
                            frontend_bitwise_right_shift ]> {
  
  let cppNamespace = "::frontend";
  let genSpecializedAttr = 0;
}

def BitwiseEnumAttr : EnumAttr<frontendDialect, BitwiseEnum, "bitwise_enum">{
      let valueType = I32;
      let cppNamespace = "::frontend";
      let assemblyFormat = "`<` $value `>`";
}

def LogicalEnum : I32EnumAttr<"LogicalEnum", "an enum", [
                            frontend_logical_and,
                            frontend_logical_or,
                            frontend_logical_not ]> {
  
  let cppNamespace = "::frontend";
  let genSpecializedAttr = 0;
}

def LogicalEnumAttr : EnumAttr<frontendDialect, LogicalEnum, "logical_enum">{
      let valueType = I32;
      let cppNamespace = "::frontend";
      let assemblyFormat = "`<` $value `>`";
}

def RelationalEnum : I32EnumAttr<"RelationalEnum", "an enum", [
                            frontend_relational_lt,
                            frontend_relational_le,
                            frontend_relational_gt,
                            frontend_relational_ge,
                            frontend_relational_eq,
                            frontend_relational_ne ]> {
  
  let cppNamespace = "::frontend";
  let genSpecializedAttr = 0;
}

def RelationalEnumAttr : EnumAttr<frontendDialect, RelationalEnum, "relational_enum">{
      let valueType = I32;
      let cppNamespace = "::frontend";
      let assemblyFormat = "`<` $value `>`";
}



def frontend_unary_arith: frontendOp<"unary_arith_op"> 
{
    let arguments = (ins AnyTensor:$src, UnaryArithEnumAttr:$arith_type);
    let results = (outs AnyTensor:$dest);
    let assemblyFormat = "$arith_type `(`$src`)` attr-dict `:` type($src) `to` type($dest)";
}


def frontend_binary_arith: frontendOp<"binary_arith_op"> 
{
    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, BinaryArithEnumAttr:$arith_type);
    let results = (outs AnyTensor:$dest);
    let assemblyFormat = "$arith_type `(`$lhs`,`$rhs`)` attr-dict `:` type($lhs)`,` type($rhs) `to` type($dest)";
}

def frontend_reduction: frontendOp<"reduction_op"> 
{
    let arguments = (ins AnyTensor:$src, ReductionEnumAttr:$reduction_type);
    let results = (outs AnyTensor:$dest);
    let assemblyFormat = "$reduction_type `(`$src`)` attr-dict `:` type($src) `to` type($dest)";
}

def frontend_bitwise: frontendOp<"bitwise_op"> 
{
    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, BitwiseEnumAttr:$bitwise_type);
    let results = (outs AnyTensor:$dest);
    let assemblyFormat = "$bitwise_type `(`$lhs`,`$rhs`)` attr-dict `:` type($lhs)`,` type($rhs) `to` type($dest)";
}

def frontend_logical: frontendOp<"logical_op"> 
{
    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, LogicalEnumAttr:$logical_type);
    let results = (outs AnyTensor:$dest);
    let assemblyFormat = "$logical_type `(`$lhs`,`$rhs`)` attr-dict `:` type($lhs)`,` type($rhs) `to` type($dest)";
}

def frontend_relational: frontendOp<"relational_op"> 
{
    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, RelationalEnumAttr:$relational_type);
    let results = (outs AnyTensor:$dest);
    let assemblyFormat = "$relational_type `(`$lhs`,`$rhs`)` attr-dict `:` type($lhs)`,` type($rhs) `to` type($dest)";
}


#endif // VKML_DIALECT_TD
