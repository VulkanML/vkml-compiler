#ifndef COMPILER_PASSES_TD
#define COMPILER_PASSES_TD

include "mlir/Pass/PassBase.td"

def lowerFrontendToMLIR : Pass<"lower-frontend-to-mlir"> {
  let summary = "Convert frontend dialect to MLIR dialect";
  let constructor = "frontend::createLowerFrontendToMLIRPass()";
  let options = [
  ];
}

def lowerFrontendToBackend : Pass<"lower-frontend-to-backend"> {
  let summary = "Convert frontend dialect to Comp dialect";
  let constructor = "frontend::createLowerFrontendToBackendPass()";
  let options = [
  ];
}

#endif // COMPILER_PASSES_TD