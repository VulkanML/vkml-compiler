#ifndef VKML_PASSES_TD
#define VKML_PASSES_TD

include "mlir/Pass/PassBase.td"

def convertVkmlToCompPass : Pass<"convert-vkml-to-comp">{

	let summary = "Convert VKML dialect to Comp dialect";
	let constructor = "frontend::createConvertVkmlToCompPass()";
	let options = [
		Option<"name", "name", "std::string", "", "help">
	];
}

def convertCompToRuntimePass : Pass<"convert-comp-to-runtime">{
	let summary = "Convert Comp dialect to Runtime dialect";
	let constructor = "runtime::createConvertCompToRuntimePass()";
	let options = [
		Option<"name", "name", "std::string", "", "help">
	];
}