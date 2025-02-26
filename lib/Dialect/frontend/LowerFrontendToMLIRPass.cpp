#include "mlir/Pass/Pass.h"
#include "frontend/Passes.h"

using namespace mlir;
using namespace frontend;

namespace {
struct LowerFrontendToMLIRPass : public PassWrapper<LowerFrontendToMLIRPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // Implement the pass logic here
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> frontend::createLowerFrontendToMLIRPass() {
  return std::make_unique<LowerFrontendToMLIRPass>();
}
