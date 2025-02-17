#include "frontend/Passes.h"
#include "mlir/Pass/PassManager.h"

// ...existing code...

void createFrontendPipeline(mlir::PassManager &pm) {
  pm.addPass(frontend::createLowerFrontendToMLIRPass());
  pm.addPass(frontend::createLowerFrontendToBackendPass());
}

// ...existing code...

int main(int argc, char **argv) {
  // ...existing code...
  mlir::PassManager pm(&context);
  createFrontendPipeline(pm);
  // ...existing code...
}
