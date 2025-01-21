set(LLVM_TARGET_DEFINITIONS pass.td)
mlir_tablegen(pass.h.inc -gen-pass-decls -name VKML)
