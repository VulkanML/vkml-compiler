#include <mlir/Pass/Pass.h>
#include <mlir/IR/PatternMatch.h>
#include "mlir/Transforms/DialectConversion.h"

#include "frontend.h"
#include "frontend_passes.h"

using namespace mlir;

namespace {
struct LowerFrontendToMLIRPass : public PassWrapper<LowerFrontendToMLIRPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
};
} // end anonymous namespace

void LowerFrontendToMLIRPass::runOnOperation() {
  // Define the conversion target
    ConversionTarget target(getContext());
    target.addIllegalDialect<frontend::frontendDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<tensor::TensorDialect>();

  // Define the type converter
    TypeConverter typeConverter;

  // Define the rewrite patterns
    OwningRewritePatternList patterns(&getContext());
  
  // Add your custom rewrite patterns here
  // patterns.insert<YourCustomPattern>(&getContext());

  // Apply the conversion
    if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}	

std::unique_ptr<Pass> frontend::createLowerFrontendToMLIRPass() {
  return std::make_unique<LowerFrontendToMLIRPass>();
}

// Register the pass
static PassRegistration<LowerFrontendToMLIRPass> pass("lower-frontend-to-mlir", "Convert frontend dialect to MLIR dialect");

