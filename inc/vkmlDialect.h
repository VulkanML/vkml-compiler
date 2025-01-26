
#ifndef VKML_DIALECT_COMPILER_H
#define VKML_DIALECT_COMPILER_H
#include <mlir/IR/Types.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>

#include <mlir/IR/BuiltinDialect.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>


#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/Transforms/Passes.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVEnums.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVAttributes.h>
#include <mlir/Target/SPIRV/Target.h>
// #include "Dialect/vkml/IR/vkmlDialect.h.inc"

namespace {
	using namespace mlir;

	static spirv::TargetEnvAttr registerTargetEnv(MLIRContext* ctx, uint32_t device_id, uint32_t vendor_id, uint32_t device_typ_id, const std::vector<uint32_t>& resource_limits,
		const std::vector<std::string>& capabilities, const std::vector<std::string>& extensions) {

		Builder builder(ctx);
		auto ver = spirv::VerCapExtAttr::get(
			spirv::Version::V_1_6,
			{
				spirv::Capability::Shader,
			},
				{
				   spirv::Extension::SPV_KHR_storage_buffer_storage_class,
				   spirv::Extension::SPV_KHR_cooperative_matrix,
				   spirv::Extension::SPV_KHR_8bit_storage,
				   spirv::Extension::SPV_KHR_16bit_storage
				},
			ctx);
		int max_compute_shared_memory_size = 1024;
		int max_compute_workgroup_invocations = 1024;
		llvm::SmallVector<Attribute> max_compute_workgroup_size;
		for (auto i : { 1024, 1024, 64 })
			max_compute_workgroup_size.push_back(builder.getI32IntegerAttr(i));
		int subgroup_size = 64;
		int min_subgroup_size = 32;
		int max_subgroup_size = 128;



		auto cooperative_matrix_properties_khr = ArrayAttr::get(ctx, {});
		auto cooperative_matrix_properties_nv = ArrayAttr::get(ctx, {});


		auto resource = spirv::ResourceLimitsAttr::get(ctx,
			max_compute_shared_memory_size,
			max_compute_workgroup_invocations,
			builder.getArrayAttr(max_compute_workgroup_size),
			subgroup_size,
			min_subgroup_size,
			max_subgroup_size,
			cooperative_matrix_properties_khr,
			cooperative_matrix_properties_nv
		);

		return spirv::TargetEnvAttr::get(
			ver,
			resource,
			spirv::ClientAPI::Vulkan,
			spirv::Vendor::Unknown,
			spirv::DeviceType::DiscreteGPU,
			device_id
		);
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