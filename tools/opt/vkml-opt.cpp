#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"


int main(int argc, char** argv) {
	mlir::DialectRegistry registry;

	return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "vkml-opt", registry));
}