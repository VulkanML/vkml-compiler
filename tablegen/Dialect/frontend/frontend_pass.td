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
  let summary = "Convert frontend dialect to backend dialect";
  let constructor = "frontend::createLowerFrontendToBackendPass()";
  let options = [
  ];
}

def lowerFrontendToMath : Pass<"lower-frontend-to-math"> {
  let summary = "Convert frontend dialect to Math dialect";
  let constructor = "frontend::createLowerFrontendToMathPass()";
  let options = [
  ];
}

def lowerFrontendToTosa : Pass<"lower-frontend-to-tosa"> {
  let summary = "Convert frontend dialect to Tosa dialect";
  let constructor = "frontend::createLowerFrontendToTosaPass()";
  let options = [
  ];
}

#endif // COMPILER_PASSES_TD