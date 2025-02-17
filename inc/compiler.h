#ifndef VKML_COMPILER_H
#define VKML_COMPILER_H

#include "frontend.h"
#include "backend.h"

#include <llvm/ADT/TypeSwitch.h>

#include <mlir/IR/Types.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassOptions.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>

#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Target/SPIRV/Target.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include "mlir/Target/SPIRV/SPIRVBinaryUtils.h"

#include <mlir/Dialect/Tensor/IR/Tensor.h>

//#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>

//#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
//#include <mlir/Dialect/SPIRV/Transforms/Passes.h>
//#include "mlir/Target/SPIRV/Serialization.h"
//#include "mlir/Target/SPIRV/Deserialization.h"
//
//#include <mlir/Target/SPIRV/Target.h>



namespace compiler {

	mlir::spirv::TargetEnvAttr registerTargetEnv(
		mlir::MLIRContext* ctx,
		uint32_t device_id,
		uint32_t vendor_id,
		uint32_t device_type_id,
		const std::vector<uint32_t>& resource_limits,
		const std::vector<std::string>& capabilities,
		const std::vector<std::string>& extensions
	); 

	mlir::Type toMLIRType(mlir::MLIRContext* ctx, uint32_t type);

class Compiler {
	mlir::MLIRContext ctx;
	mlir::ModuleOp global_module;
	mlir::func::FuncOp main_func_op;
	mlir::Block* entry_block;

	mlir::OpBuilder global_builder;
	std::vector<mlir::ModuleOp> device_modules;
	std::map<std::string, mlir::Operation*> srcs;
	mlir::PassManager pm;



public:
	~Compiler() {}
	
	void addDevice(uint32_t device_id, uint32_t vendor_id, uint32_t device_type_id, const std::vector<uint32_t>& resource_limits, const std::vector<std::string>& capabilities, const std::vector<std::string>& extensions)
	{
		auto target_env = registerTargetEnv(&ctx, device_id, vendor_id, device_type_id, resource_limits, capabilities, extensions);
		auto device_module = global_builder.create<mlir::ModuleOp>(global_builder.getUnknownLoc(), "device_" + std::to_string(device_id));
		device_module->setAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName(), global_builder.getUnitAttr());
		device_module->setAttr(mlir::spirv::getTargetEnvAttrName(), target_env);
		this->device_modules.push_back(device_module);
	}

	void addTensor(std::string name, const std::vector<int64_t>& shape, uint32_t type, uint32_t aligninment) {
        mlir::OpBuilder builder(main_func_op.getBody());
		builder.setInsertionPointToStart(entry_block);
		mlir::MemRefType memrefType{};		
		
		mlir::RankedTensorType tensorType = mlir::RankedTensorType::get(shape, toMLIRType(&ctx, type));
		auto src_op = builder.create<mlir::tensor::EmptyOp>(builder.getUnknownLoc(), tensorType, mlir::ValueRange{});
		srcs[name] = src_op.getOperation();
	}

	void addOp(size_t operator_type, std::string op_name, std::string input_name) {
		mlir::OpBuilder builder(main_func_op.getBody());
		builder.setInsertionPointAfterValue(srcs[input_name]->getResult(0));
		mlir::Operation* input_op = srcs[input_name];
		switch (operator_type) {
		case 0:
			auto t = mlir::dyn_cast<mlir::tensor::EmptyOp>(input_op).getType();
			auto abs_op = builder.create<frontend::abs>(builder.getUnknownLoc(), t, srcs[input_name]->getResult(0));
			srcs[op_name] = abs_op.getOperation();
			break;
		};

	}
	
	void dump() {
		global_module.dump();
	}

	Compiler() : ctx(), global_builder(&ctx), pm(&ctx)
	{
		ctx.loadDialect<mlir::tensor::TensorDialect, mlir::gpu::GPUDialect, mlir::func::FuncDialect, mlir::math::MathDialect, frontend::frontendDialect>();
		mlir::spirv::registerSPIRVTargetInterfaceExternalModels(ctx);
		this->initialize();
	}

	void initialize() {
		this->global_module = global_builder.create<mlir::ModuleOp>(global_builder.getUnknownLoc());
		main_func_op = global_builder.create<mlir::func::FuncOp>(global_builder.getUnknownLoc(), "main", global_builder.getFunctionType({}, {}));
		entry_block = main_func_op.addEntryBlock();
		
		mlir::OpBuilder builder(main_func_op.getBody());
		auto returnOp = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());	
		this->global_module.push_back(main_func_op.getOperation());
	}

	void run() {
		std::vector<mlir::Operation*> device_modules;

		global_module->walk([&](mlir::Operation* op) {
			auto mop = mlir::dyn_cast_or_null<mlir::ModuleOp>(op);
			if (mop != nullptr)
				device_modules.push_back(op);
			}
		);

		for (auto* dMod : device_modules) {
			pm.run(dMod);
		}
	}
};




}

#endif