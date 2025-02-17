#ifndef COMPILER_PASSES_TD
#define COMPILER_PASSES_TD

include "mlir/Pass/PassBase.td"

def lowerFrontendToBackend : Pass<"lower-frontend-to-backend">{
	let summary = "Convert VKML dialect to Comp dialect";
	let constructor = "frontend::createConvertVkmlToCompPass()";
	let options = [
		Option<"name", "name", "std::string", "", "help">
	];
}

#endif // COMPILER_PASSES_TD