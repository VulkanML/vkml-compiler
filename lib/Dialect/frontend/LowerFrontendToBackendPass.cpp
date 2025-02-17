#include <mlir/Pass/Pass.h>
#include <mlir/IR/PatternMatch.h>
#include "mlir/Transforms/DialectConversion.h"

#include "frontend.h"
#include "frontend_passes.h"

using namespace mlir;
using namespace frontend;

namespace {
    struct LowerFrontendToBackendPass : public PassWrapper<LowerFrontendToBackendPass, OperationPass<mlir::ModuleOp>> {
        void runOnOperation() override;
    };
} // end anonymous namespace

std::unique_ptr<Pass> frontend::createLowerFrontendToBackendPass() {
    return std::make_unique<LowerFrontendToBackendPass>();
}


void LowerFrontendToBackendPass::runOnOperation() {
    // Define the conversion target
    ConversionTarget target(getContext());
    target.addIllegalDialect<frontend::frontendDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::tensor::TensorDialect>();

    // Define the type converter
    TypeConverter typeConverter;

    // Define the rewrite patterns
    mlir::RewritePatternSet patterns(&getContext());

    // Add your custom rewrite patterns here
    // patterns.insert<YourCustomPattern>(&getContext());

    // Apply the conversion
    if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}

// Register the pass
static PassRegistration<LowerFrontendToBackendPass> pass("lower-frontend-to-mlir", "Convert frontend dialect to MLIR dialect");

