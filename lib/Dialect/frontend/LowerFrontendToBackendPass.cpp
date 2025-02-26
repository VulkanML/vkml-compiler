#include "mlir/Pass/Pass.h"
#include "frontend/Passes.h"

using namespace mlir;
using namespace frontend;

namespace {
struct LowerFrontendToBackendPass : public PassWrapper<LowerFrontendToBackendPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // Implement the pass logic here
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> frontend::createLowerFrontendToBackendPass() {
  return std::make_unique<LowerFrontendToBackendPass>();
}
