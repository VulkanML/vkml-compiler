
#ifndef VKML_DIALECT_COMPILER_H
#define VKML_DIALECT_COMPILER_H
#include <mlir/IR/Types.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/BuiltinDialect.h>

#include <mlir/Pass/PassManager.h>

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>

#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/Transforms/Passes.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Target/SPIRV/Deserialization.h"
#include "mlir/Target/SPIRV/SPIRVBinaryUtils.h"
#include <mlir/Target/SPIRV/Target.h>

#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h>

#include "Dialect/vkml/vkml.h.inc"
#include "Dialect/vkml/vkmlDialect.h.inc"
#include "Dialect/vkml/vkmlTypes.h.inc"

namespace {


	static mlir::spirv::TargetEnvAttr registerTargetEnv(
		mlir::MLIRContext* ctx,
		uint32_t device_id,
		uint32_t vendor_id,
		uint32_t device_type_id,
		const std::vector<uint32_t>& resource_limits,
		const std::vector<std::string>& capabilities,
		const std::vector<std::string>& extensions 
	) {
		std::vector<mlir::spirv::Capability> caps;
		std::vector<mlir::spirv::Extension> exts;
		caps.push_back(mlir::spirv::Capability::Shader);

		for (auto& cap : capabilities) 
			caps.emplace_back(mlir::spirv::symbolizeCapability(cap).value());
		
		for (auto& ext : extensions) {
			auto e = mlir::spirv::symbolizeExtension(ext);
			if(e.has_value())
				exts.emplace_back(e.value());
		}

		mlir::Builder builder(ctx);

		auto cooperative_matrix_properties_khr = mlir::ArrayAttr::get(ctx, {});
		auto cooperative_matrix_properties_nv = mlir::ArrayAttr::get(ctx, {});

		llvm::SmallVector<mlir::Attribute> maxComputeWorkGroupSize;
		maxComputeWorkGroupSize.emplace_back(builder.getI32IntegerAttr(resource_limits[2]));
		maxComputeWorkGroupSize.emplace_back(builder.getI32IntegerAttr(resource_limits[3]));
		maxComputeWorkGroupSize.emplace_back(builder.getI32IntegerAttr(resource_limits[4]));

		return mlir::spirv::TargetEnvAttr::get(
			mlir::spirv::VerCapExtAttr::get(mlir::spirv::Version::V_1_6, caps, exts, ctx),
			mlir::spirv::ResourceLimitsAttr::get(ctx,
				resource_limits[0],
				resource_limits[1],
				builder.getArrayAttr(maxComputeWorkGroupSize),
				resource_limits[5],
				resource_limits[6],
				resource_limits[7],
				cooperative_matrix_properties_khr,
				cooperative_matrix_properties_nv
			),
			mlir::spirv::ClientAPI::Vulkan,
			(mlir::spirv::Vendor)vendor_id,
			(mlir::spirv::DeviceType)device_type_id,
			device_id
		);
	}

	static void defineVKMLDialectPasses(mlir::PassManager* pm) {
		pm->addPass(mlir::createGpuKernelOutliningPass());
		//pm->addPass(mlir::memref::createFoldMemRefAliasOpsPass());
		//pm->addPass(mlir::createConvertGPUToSPIRVPass(/*mapMemorySpace=*/true));

		//mlir::OpPassManager& spirPM = pm->nest<mlir::spirv::ModuleOp>();
		//spirPM.addPass(mlir::spirv::createSPIRVLowerABIAttributesPass());
		//spirPM.addPass(mlir::spirv::createSPIRVUpdateVCEPass());
	}

	

}


//
//inline ModuleOp initalize(MLIRContext& ctx) {
//    ctx.loadDialect<gpu::GPUDialect, func::FuncDialect, arith::ArithDialect, memref::MemRefDialect, spirv::SPIRVDialect>();
//    OpBuilder builder(&ctx);
//    auto mod = builder.create<ModuleOp>(builder.getUnknownLoc());
//    spirv::registerSPIRVTargetInterfaceExternalModels(ctx);
//    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), builder.getUnitAttr());
//    mod->setAttr(spirv::getTargetEnvAttrName(), registerTargetEnv(&ctx));
//    return mod;
//}
//inline gpu::GPUModuleOp createGpuModule(ModuleOp global_mod) {
//    OpBuilder builder(global_mod.getContext());
//    auto ctx = global_mod.getContext();
//    auto mod = builder.create<gpu::GPUModuleOp>(builder.getUnknownLoc(), "kernels");
//    mod->setAttr(spirv::getTargetEnvAttrName(), registerTargetEnv(ctx));
//    return mod;
//}
//}



#endif