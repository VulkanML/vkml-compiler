#include "vkml_compiler.h"


namespace vkml_compiler {
	compiler::compiler() : ctx(), global_builder(&ctx)
	{
		ctx.loadDialect<gpu::GPUDialect, spirv::SPIRVDialect>();
		spirv::registerSPIRVTargetInterfaceExternalModels(ctx);
		this->initialize();
	}
	void compiler::initialize()
	{
		

		this->global_module = global_builder.create<mlir::ModuleOp>(global_builder.getUnknownLoc());
	}

	void compiler::addDevice(uint32_t device_id, uint32_t vendor_id, uint32_t device_type_id, const std::vector<std::string>& capabilities, const std::vector<std::string>& extensions)
	{

		auto target_env = registerTargetEnv(&ctx, device_id, vendor_id, device_type_id, {}, capabilities, extensions);
		auto device_module = global_builder.create<mlir::ModuleOp>(global_builder.getUnknownLoc(), "device_" + std::to_string(device_id));
		device_module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), global_builder.getUnitAttr());
		device_module->setAttr(spirv::getTargetEnvAttrName(), target_env);

		//this->device_modules.push_back();
	}



}