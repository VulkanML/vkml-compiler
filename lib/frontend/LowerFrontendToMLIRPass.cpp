#include <mlir/Pass/Pass.h>
#include <mlir/IR/PatternMatch.h>
#include "mlir/Transforms/DialectConversion.h"

#include "frontend.h"
#include "frontend_passes.h"

using namespace mlir;

namespace frontend {

  struct LowerFrontendToMLIRPass : public PassWrapper<LowerFrontendToMLIRPass, OperationPass<ModuleOp>> {
      LowerFrontendToMLIRPass() = default;
      void runOnOperation() override;
      void getDependentDialects(DialectRegistry& registry) const final {
          registry.insert<mlir::func::FuncDialect>();
          registry.insert<mlir::arith::ArithDialect>();
          registry.insert<mlir::tensor::TensorDialect>();
      }
  };

  void LowerFrontendToMLIRPass::runOnOperation() {
    // Define the conversion target
      ConversionTarget target(getContext());
      // target.addIllegalDialect<frontend::frontendDialect>();
      target.addLegalDialect<arith::ArithDialect>();
      target.addLegalDialect<func::FuncDialect>();
      target.addLegalDialect<tensor::TensorDialect>();

    // Define the type converter
      TypeConverter typeConverter;

    // Define the rewrite patterns
      RewritePatternSet patterns(&getContext());
    
    // Add your custom rewrite patterns here
    // patterns.insert<YourCustomPattern>(&getContext());

    // Apply the conversion
      if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
          signalPassFailure();
  }	

    
  std::unique_ptr<Pass> frontend::createLowerFrontendToMLIRPass() {
    return std::make_unique<frontend::LowerFrontendToMLIRPass>();
  }
} // end anonymous namespace


