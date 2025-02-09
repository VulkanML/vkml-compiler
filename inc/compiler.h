#ifndef VKML_COMPILER_H
#define VKML_COMPILER_H

#include "vkmlDialect.h"

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



namespace compiler {

class Tensor {
public:
	Tensor() {}
	~Tensor() {}
};

class Compiler {
	mlir::MLIRContext ctx;
	mlir::ModuleOp global_module;
	mlir::func::FuncOp main_func_op;
	mlir::Block* entry_block;

	mlir::OpBuilder global_builder;
	std::vector<mlir::ModuleOp> device_modules;
	std::vector<mlir::Operation*> stack;
	mlir::PassManager pm;

public:
	~Compiler() {}
	
	void addDevice(uint32_t device_id, uint32_t vendor_id, uint32_t device_type_id, const std::vector<uint32_t>& resource_limits, const std::vector<std::string>& capabilities, const std::vector<std::string>& extensions)
	{
		auto target_env = registerTargetEnv(&ctx, device_id, vendor_id, device_type_id, resource_limits, capabilities, extensions);
		auto device_module = global_builder.create<mlir::ModuleOp>(global_builder.getUnknownLoc(), "device_" + std::to_string(device_id));
		device_module->setAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName(), global_builder.getUnitAttr());
		device_module->setAttr(mlir::spirv::getTargetEnvAttrName(), target_env);
		device_modules.push_back(device_module);
	}

	mlir::memref::AllocOp addTensor(std::string name, std::vector<int64_t>& shape, uint32_t type, uint32_t aligninment) {
        mlir::OpBuilder builder(main_func_op.getBody());
		builder.setInsertionPointToStart(entry_block);
		mlir::MemRefType memrefType{};

		switch (type) {
		case 1:
			memrefType = mlir::MemRefType::get(shape, builder.getIntegerType(32, false));
			break;
		case 2:
			memrefType = mlir::MemRefType::get(shape, builder.getIntegerType(8, true));
			break;
		case 3:
			memrefType = mlir::MemRefType::get(shape, builder.getIntegerType(8, false));	
			break;
		case 4:
			memrefType = mlir::MemRefType::get(shape, builder.getIntegerType(16, true));
			break;
		case 5:
			memrefType = mlir::MemRefType::get(shape, builder.getIntegerType(16, false));
			break;
		case 6:
			memrefType = mlir::MemRefType::get(shape, builder.getIntegerType(32, true));
			break;
		case 7:
			memrefType = mlir::MemRefType::get(shape, builder.getIntegerType(32, false));
			break;
		case 8:
			memrefType = mlir::MemRefType::get(shape, builder.getIntegerType(64, true));
			break;
		case 9:
			memrefType = mlir::MemRefType::get(shape, builder.getIntegerType(64, false));
			break;
		case 10:
			memrefType = mlir::MemRefType::get(shape, builder.getIntegerType(64, true));
			break;
		case 11:
			memrefType = mlir::MemRefType::get(shape, builder.getIntegerType(64, false));
			break;
		case 12:
			memrefType = mlir::MemRefType::get(shape, builder.getF16Type());
			break;
		case 13:
			memrefType = mlir::MemRefType::get(shape, builder.getF32Type());
			break;
		case 14:
			memrefType = mlir::MemRefType::get(shape, builder.getF64Type());
			break;
		case 15:
			memrefType = mlir::MemRefType::get(shape, builder.getF128Type());
			break;
		case 0:
		default:
			memrefType = mlir::MemRefType::get(shape, builder.getNoneType());
			break;

		};

 		auto memrefOp_alloc = builder.create<mlir::memref::AllocOp>(builder.getUnknownLoc(), memrefType);
		return memrefOp_alloc;
	}
	
	void dump() {
		global_module.dump();
	}

	Compiler() : ctx(), global_builder(&ctx), pm(&ctx)
	{
		ctx.loadDialect<mlir::gpu::GPUDialect, mlir::spirv::SPIRVDialect, mlir::func::FuncDialect>();
		mlir::spirv::registerSPIRVTargetInterfaceExternalModels(ctx);
		this->initialize();
	}

	void initialize() {
		this->global_module = global_builder.create<mlir::ModuleOp>(global_builder.getUnknownLoc());
		main_func_op = global_builder.create<mlir::func::FuncOp>(global_builder.getUnknownLoc(), "main", global_builder.getFunctionType({}, {}));
		entry_block = main_func_op.addEntryBlock();
		
		mlir::OpBuilder builder(main_func_op.getBody());
		auto returnOp = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());			
	}

	void run() {
		std::vector<mlir::Operation*> device_modules;

		global_module->walk([&](mlir::Operation* op) {
			auto mop = mlir::dyn_cast_or_null<mlir::ModuleOp>(op);
			if (mop != nullptr)
				device_modules.push_back(mop.getOperation());
			}
		);

		for (auto* dMod : device_modules) {
			pm.run(dMod);
		}
	}
};




}

#endif