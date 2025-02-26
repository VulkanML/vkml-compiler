#ifndef FRONTEND_PASSES_H
#define FRONTEND_PASSES_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace frontend {
std::unique_ptr<mlir::Pass> createLowerFrontendToMLIRPass();
std::unique_ptr<mlir::Pass> createLowerFrontendToBackendPass();
} // namespace frontend

#endif // FRONTEND_PASSES_H
