#include "compiler.h"



namespace compiler {
	mlir::spirv::TargetEnvAttr registerTargetEnv(
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
			if (e.has_value())
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

    mlir::Type toMLIRType(mlir::MLIRContext* ctx, uint32_t type) {
		switch (type) {
		case 1:
			return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Unsigned);
		case 2:
			return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signed);
		case 3:
			return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);
		case 4:
			return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Signed);
		case 5:
			return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Unsigned);
		case 6:
			return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
		case 7:
			return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Unsigned);
		case 8:
			return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
		case 9:
			return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
		case 10:
			return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
		case 11:
			return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
		case 12:
			return mlir::Float16Type::get(ctx);
		case 13:
			return mlir::Float32Type::get(ctx);
		case 14:
			return mlir::Float64Type::get(ctx);
		case 15:
			return mlir::Float128Type::get(ctx);
		case 0:
		default:
			return mlir::NoneType::get(ctx);
		};
	}
}