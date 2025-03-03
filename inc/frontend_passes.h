#ifndef FRONTEND_PASSES_H
#define FRONTEND_PASSES_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

namespace frontend {

#define GEN_PASS_DECL
#include "Dialect/frontend/frontend_pass.h.inc"

std::unique_ptr<mlir::Pass> createLowerFrontendToMLIRPass();
std::unique_ptr<mlir::Pass> createLowerFrontendToBackendPass();
std::unique_ptr<mlir::Pass> createLowerFrontendToMathPass();
std::unique_ptr<mlir::Pass> createLowerFrontendToTosaPass();

#define GEN_PASS_REGISTRATION
#include "Dialect/frontend/frontend_pass.h.inc"


} // namespace frontend

#endif // FRONTEND_PASSES_H
