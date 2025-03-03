#ifndef VKML_COMPILER_H
#define VKML_COMPILER_H

#include "frontend.h"
#include "backend.h"
#include <iostream>
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

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>

#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

#include <mlir/Dialect/SCF/IR/SCF.h>  // Add SCF dialect for loops

#include <mlir/Conversion/TosaToLinalg/TosaToLinalg.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>

// #include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
// #include <mlir/Dialect/SPIRV/Transforms/Passes.h>
// #include "mlir/Target/SPIRV/Serialization.h"
// #include "mlir/Target/SPIRV/Deserialization.h"
//
// #include <mlir/Target/SPIRV/Target.h>
// #include "mlir/Dialect/MemRef/Transforms/Passes.h"

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
    mlir::func::ReturnOp main_return_op;
    mlir::OpBuilder main_func_builder;
    mlir::Block* entry_block;

    mlir::OpBuilder global_builder;
    std::vector<mlir::ModuleOp> device_modules;
    std::map<std::string, mlir::Operation*> srcs;
    mlir::PassManager pm;

public:
    Compiler() : ctx(), global_builder(&ctx), pm(&ctx), main_func_builder(&ctx) {
        ctx.loadDialect<mlir::tensor::TensorDialect,
                       mlir::gpu::GPUDialect, 
                       mlir::func::FuncDialect, 
                       mlir::tosa::TosaDialect,
                       mlir::linalg::LinalgDialect,
                       mlir::scf::SCFDialect,
                       frontend::frontendDialect>();
        mlir::spirv::registerSPIRVTargetInterfaceExternalModels(ctx);
        this->initialize();
        this->setfrontendPipeline();
    }

    void initialize() {
        this->global_module = global_builder.create<mlir::ModuleOp>(global_builder.getUnknownLoc());
        main_func_op = global_builder.create<mlir::func::FuncOp>(global_builder.getUnknownLoc(), "main", global_builder.getFunctionType({}, {}));
        entry_block = main_func_op.addEntryBlock();
        main_func_builder.setInsertionPointToStart(entry_block);
        main_return_op = main_func_builder.create<mlir::func::ReturnOp>(main_func_builder.getUnknownLoc());
        this->global_module.push_back(main_func_op.getOperation());
    }

    void setfrontendPipeline() {
        // Convert frontend dialect to TOSA
        //pm.addPass(frontend::createLowerFrontendToTosaPass());
        
        // Convert TOSA to Linalg
        //pm.addNestedPass<func::FuncOp>(mlir::tosa::createTosaToLinalg());

       // pm.addPass(frontend::createLowerFrontendToMathPass());        
       
    }

    void runfrontendPipeline() {
        if (failed(pm.run(global_module))) {
            std::cout << "Error: frontend pipeline failed\n";
            return;
        }
    }

    void addDevice(uint32_t device_id, uint32_t vendor_id, uint32_t device_type_id, const std::vector<uint32_t>& resource_limits, const std::vector<std::string>& capabilities, const std::vector<std::string>& extensions) {
        auto target_env = registerTargetEnv(&ctx, device_id, vendor_id, device_type_id, resource_limits, capabilities, extensions);
        auto device_module = global_builder.create<mlir::ModuleOp>(global_builder.getUnknownLoc(), "device_" + std::to_string(device_id));
        device_module->setAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName(), global_builder.getUnitAttr());
        device_module->setAttr(mlir::spirv::getTargetEnvAttrName(), target_env);
        this->device_modules.push_back(device_module);
		this->global_module.push_back(device_module);

    }

    void addTensor(std::string name, const std::vector<int64_t>& shape, uint32_t type, uint32_t aligninment) {
        main_func_builder.setInsertionPoint(main_return_op);
        mlir::RankedTensorType tensorType = mlir::RankedTensorType::get(shape, toMLIRType(&ctx, type));
        auto src_op = main_func_builder.create<mlir::tensor::EmptyOp>(main_func_builder.getUnknownLoc(), tensorType, mlir::ValueRange{});
        srcs[name] = src_op.getOperation();
    }

    void addUnaryOp(size_t operator_type, std::string op_name, std::string input_name) {
        mlir::Operation* input_op = srcs[input_name];
        auto unary_operator_type = frontend::symbolizeUnaryArithEnum(operator_type);
        auto type = input_op->getResult(0).getType();
        if (unary_operator_type.has_value()) {
            main_func_builder.setInsertionPoint(main_return_op);
            auto ops = main_func_builder.create<frontend::unary_arith>(main_func_builder.getUnknownLoc(), type, input_op->getResult(0), frontend::UnaryArithEnumAttr::get(&ctx, unary_operator_type.value()));
            srcs[op_name] = ops.getOperation();
        } else {
            throw std::exception("Invalid operator type");
        }
    }

    void addBinaryOp(size_t operator_type, std::string op_name, std::string input_0_name, std::string input_1_name) {
        mlir::OpBuilder::InsertionGuard guard(main_func_builder);
        mlir::Operation* input_op_0 = srcs[input_0_name];
        mlir::Operation* input_op_1 = srcs[input_1_name];
        auto binary_operator_type = frontend::symbolizeBinaryArithEnum(operator_type);
        if (binary_operator_type.has_value()) {
            main_func_builder.setInsertionPoint(main_return_op);
            auto ops = main_func_builder.create<frontend::binary_arith>(main_func_builder.getUnknownLoc(), input_op_0->getResult(0).getType(), input_op_0->getResult(0), input_op_1->getResult(0), frontend::BinaryArithEnumAttr::get(&ctx, binary_operator_type.value()));
            srcs[op_name] = ops.getOperation();
        } else {
            throw std::exception("Invalid operator type");
        }
    }

    void addReductionOp(size_t operator_type, std::string op_name, std::string input_name) {
        mlir::OpBuilder::InsertionGuard guard(main_func_builder);
        mlir::Operation* input_op = srcs[input_name];
        auto reduction_operator_type = frontend::symbolizeReductionEnum(operator_type);
        if (reduction_operator_type.has_value()) {
            main_func_builder.setInsertionPoint(main_return_op);
            auto ops = main_func_builder.create<frontend::reduction>(main_func_builder.getUnknownLoc(), input_op->getResult(0).getType(), input_op->getResult(0), frontend::ReductionEnumAttr::get(&ctx, reduction_operator_type.value()));
            srcs[op_name] = ops.getOperation();
        } else {
            throw std::exception("Invalid operator type");
        }
    }

    void addBitwiseOp(size_t operator_type, std::string op_name, std::string input_0_name, std::string input_1_name) {
        mlir::OpBuilder::InsertionGuard guard(main_func_builder);
        mlir::Operation* input_op_0 = srcs[input_0_name];
        mlir::Operation* input_op_1 = srcs[input_1_name];
        auto bitwise_operator_type = frontend::symbolizeBitwiseEnum(operator_type);
        if (bitwise_operator_type.has_value()) {
            main_func_builder.setInsertionPoint(main_return_op);
            auto ops = main_func_builder.create<frontend::bitwise>(main_func_builder.getUnknownLoc(), input_op_0->getResult(0).getType(), input_op_0->getResult(0), input_op_1->getResult(0), frontend::BitwiseEnumAttr::get(&ctx, bitwise_operator_type.value()));
            srcs[op_name] = ops.getOperation();
        } else {
            throw std::exception("Invalid operator type");
        }
    }

    void addLogicalOp(size_t operator_type, std::string op_name, std::string input_0_name, std::string input_1_name) {
        mlir::OpBuilder::InsertionGuard guard(main_func_builder);
        mlir::Operation* input_op_0 = srcs[input_0_name];
        mlir::Operation* input_op_1 = srcs[input_1_name];
        auto logical_operator_type = frontend::symbolizeLogicalEnum(operator_type);
        if (logical_operator_type.has_value()) {
            main_func_builder.setInsertionPoint(main_return_op);
            auto ops = main_func_builder.create<frontend::logical>(main_func_builder.getUnknownLoc(), input_op_0->getResult(0).getType(), input_op_0->getResult(0), input_op_1->getResult(0), frontend::LogicalEnumAttr::get(&ctx, logical_operator_type.value()));
            srcs[op_name] = ops.getOperation();
        } else {
            throw std::exception("Invalid operator type");
        }
    }

    void addRelationalOp(size_t operator_type, std::string op_name, std::string input_0_name, std::string input_1_name) {
        mlir::OpBuilder::InsertionGuard guard(main_func_builder);
        mlir::Operation* input_op_0 = srcs[input_0_name];
        mlir::Operation* input_op_1 = srcs[input_1_name];
        auto relational_operator_type = frontend::symbolizeRelationalEnum(operator_type);
        if (relational_operator_type.has_value()) {
            main_func_builder.setInsertionPoint(main_return_op);
            auto ops = main_func_builder.create<frontend::relational>(main_func_builder.getUnknownLoc(), input_op_0->getResult(0).getType(), input_op_0->getResult(0), input_op_1->getResult(0), frontend::RelationalEnumAttr::get(&ctx, relational_operator_type.value()));
            srcs[op_name] = ops.getOperation();
        } else {
            throw std::exception("Invalid operator type");
        }
    }

    void dump() {
        global_module.dump();
    }

    void run() {
        std::vector<mlir::Operation*> device_modules;
        this->runfrontendPipeline();
    }
    ~Compiler()
    {

    }
};

}
#endif