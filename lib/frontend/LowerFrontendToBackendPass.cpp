#include <mlir/Pass/Pass.h>
#include <mlir/IR/PatternMatch.h>
#include "mlir/Transforms/DialectConversion.h"

#include "frontend.h"
#include "frontend_passes.h"

using namespace mlir;

namespace frontend {
    struct LowerFrontendToBackendPass : public PassWrapper<LowerFrontendToBackendPass, OperationPass<mlir::ModuleOp>> {
		LowerFrontendToBackendPass() = default;
        void runOnOperation() override;
        void getDependentDialects(DialectRegistry &registry) const final {
            registry.insert<mlir::func::FuncDialect>();
            registry.insert<mlir::arith::ArithDialect>();
            registry.insert<mlir::tensor::TensorDialect>();
        }
    };

std::unique_ptr<Pass> frontend::createLowerFrontendToBackendPass() {
    return std::make_unique<LowerFrontendToBackendPass>();
}

void LowerFrontendToBackendPass::runOnOperation() {
    // Define the conversion target
    ConversionTarget target(getContext());
    // target.addIllegalDialect<frontend::frontendDialect>();
    target.addLegalDialect<mlir::BuiltinDialect>();
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


} // end frontend namespace