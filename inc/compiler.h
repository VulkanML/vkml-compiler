#pragma once

#include <memory>
#include <stack>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"


// #include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
// #include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
// #include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
// #include "mlir/Conversion/TosaToArith/TosaToArith.h"
// #include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
// #include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"


// #include "mlir/Dialect/Linalg/Passes.h"
// #include "mlir/Dialect/Bufferization/Transforms/Passes.h"
// #include "mlir/Dialect/GPU/Transforms/Passes.h"
// #include "mlir/Dialect/Affine/Passes.h"


#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h" // canonicalizer, CSE

// Affine Pass
#include "mlir/Dialect/Affine/Passes.h"

// TOSA conversions
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToSCF/TosaToSCF.h"

// Linalg passes
#include "mlir/Dialect/Linalg/Passes.h"

// Bufferization
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

// GPU passes
#include "mlir/Dialect/GPU/Transforms/Passes.h"

// GPU to target ISA
#include "mlir/Conversion/GPUToNVVM/GPUToNVVM.h"
// #include "mlir/Conversion/GPUToROCDL/GPUToROCDL.h"

// LLVM conversions
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"

namespace vkml {

// Forward declarations
class TensorOps;
class TosaOps;

class Compiler {
private:
    struct StackEntry {
        mlir::Operation* op;
        llvm::SmallVector<mlir::Value> dependencies;
        std::string funcName; // Name of the generated function for this op
        llvm::SmallVector<mlir::Type> argTypes; // Argument types for the function
        llvm::SmallVector<mlir::Value> args; // Arguments to pass to the call

        StackEntry(mlir::Operation* operation, llvm::ArrayRef<mlir::Value> deps = {},
            const std::string& fn = "", llvm::ArrayRef<mlir::Type> argTys = {}, llvm::ArrayRef<mlir::Value> callArgs = {})
            : op(operation), dependencies(deps.begin(), deps.end()), funcName(fn), argTypes(argTys.begin(), argTys.end()), args(callArgs.begin(), callArgs.end()) {}
    };

        // Build the IR from the stack
    void buildFromStack() {
        if (opStack_->empty()) {
            printf("Warning: Operation stack is empty\n");
            return;
        }

        // Get the final operation to determine return type
        auto finalEntry = opStack_->top();
        if (!finalEntry.op || finalEntry.op->getNumResults() == 0) {
            printf("Error: final operation is null or has no results\n");
            return;
        }
        createMainFunction();

        mlir::OpBuilder builder(context_.get());
        builder.setInsertionPointToStart(&mainFunc_.getBody().front());

        // Process stack in reverse order (LIFO -> correct dependency order)
        std::vector<StackEntry> operations;
        while (!opStack_->empty()) {
            operations.push_back(opStack_->top());
            opStack_->pop();
        }
        std::reverse(operations.begin(), operations.end());

        // Map to track the result of each op for argument wiring
        std::unordered_map<mlir::Operation*, mlir::Value> opResultMap;
        mlir::Value lastResult;

        // For each op, if it needs i nput tensors, create them in main
        llvm::DenseMap<mlir::Type, mlir::Value> createdInputs;

        builder.setInsertionPointToStart(&mainFunc_.getBody().front());
        for (auto& entry : operations) {
            builder.insert(entry.op);
            lastResult = entry.op->getResult(0);
            
        }

        // Add return statement
        builder.setInsertionPointToEnd(&mainFunc_.getBody().front());
        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    }

    std::shared_ptr<mlir::MLIRContext> context_;
    mlir::ModuleOp module_;
    mlir::func::FuncOp mainFunc_;
    mlir::DialectRegistry registry_;
    std::shared_ptr<std::stack<StackEntry>> opStack_;
    bool functionCreated_;

public:
    Compiler() : context_(std::make_shared<mlir::MLIRContext>()), 
                 opStack_(std::make_shared<std::stack<StackEntry>>()),
                 functionCreated_(false) {
        // Register all required dialects
        registry_.insert<mlir::tensor::TensorDialect>();
        registry_.insert<mlir::tosa::TosaDialect>();
        registry_.insert<mlir::func::FuncDialect>();
        registry_.insert<mlir::linalg::LinalgDialect>();
        registry_.insert<mlir::arith::ArithDialect>();

        mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry_);
        mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry_);
        mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry_);

        context_->appendDialectRegistry(registry_);
        context_->loadDialect<mlir::tensor::TensorDialect, 
                             mlir::tosa::TosaDialect, 
                             mlir::func::FuncDialect, 
                             mlir::linalg::LinalgDialect,
                             mlir::arith::ArithDialect>();

        // Register BufferizableOpInterface external models for tensor and linalg dialects

        // Create module
        module_ = mlir::ModuleOp::create(mlir::UnknownLoc::get(context_.get()));
    }

    Compiler(const Compiler&) = delete;
    Compiler& operator=(const Compiler&) = delete;

    mlir::MLIRContext* getContext() const { 
        return context_.get(); 
    }

    mlir::ModuleOp getModule() const { 
        return module_; 
    }

    mlir::func::FuncOp getFunc() { 
        createMainFunction();
        return mainFunc_; 
    }

    std::shared_ptr<std::stack<StackEntry>> getOpStack() const {
        return opStack_;
    }

    void dump() {
        module_.dump();
    }

    // Create the main function when we have the final operation
    void createMainFunction() {
        if (functionCreated_) return;
        
        mlir::OpBuilder builder(context_.get());
        builder.setInsertionPointToEnd(module_.getBody());
        
        auto funcType = builder.getFunctionType({}, {});
        mainFunc_ = builder.create<mlir::func::FuncOp>( builder.getUnknownLoc(), "main", funcType);
        mainFunc_.addEntryBlock();
        
        functionCreated_ = true;
    }


    void runPasses() {
        buildFromStack();

        module_.dump();

        // Create PassManager and configure passes
        mlir::PassManager pm(context_.get());

        // 1. Canonicalize initial IR
        pm.addPass(mlir::createCanonicalizerPass());

        // 2. Lower TOSA to Linalg/Arith/SCF
        pm.nest<mlir::func::FuncOp>().addPass(mlir::tosa::createTosaToLinalg());
        pm.nest<mlir::func::FuncOp>().addPass(mlir::tosa::createTosaToArith());
        pm.nest<mlir::func::FuncOp>().addPass(mlir::tosa::createTosaToSCF());
        pm.addPass(mlir::createCanonicalizerPass());

        // 3. Bufferization (tensors → memrefs)
        mlir::bufferization::OneShotBufferizationOptions opts;
        opts.bufferizeFunctionBoundaries = true;
        opts.setFunctionBoundaryTypeConversion(mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
        pm.addPass(mlir::bufferization::createOneShotBufferizePass(opts));
        pm.addPass(mlir::createCanonicalizerPass());


        // 4. Lower Linalg to affine loops
        pm.nest<mlir::func::FuncOp>().addPass(mlir::createConvertLinalgToParallelLoopsPass());
        pm.nest<mlir::func::FuncOp>().addPass(mlir::affine::createAffineLoopInvariantCodeMotionPass());


        // 6. Lower parallel loops to GPU
        pm.nest<mlir::func::FuncOp>().addPass(mlir::createGpuMapParallelLoopsPass());
        pm.nest<mlir::func::FuncOp>().addPass(mlir::createParallelLoopToGpuPass());
        pm.addPass(mlir::createGpuKernelOutliningPass());

        // // 7. Lower remaining affine ops
        // pm.addPass(mlir::createLowerAffinePass());

        if (mlir::failed(pm.run(module_))) {
            printf("Failed to run passes\n");
        }

        module_.dump();
    }

    friend class TensorOps;
    friend class TosaOps;
};

class TensorOps {
private:
    std::shared_ptr<mlir::OpBuilder> builder_;
    std::shared_ptr<std::stack<Compiler::StackEntry>> opStack_;

public:
    TensorOps(Compiler& compiler) {
        builder_ = std::make_shared<mlir::OpBuilder>(compiler.getContext());
        opStack_ = compiler.getOpStack();
    }

    mlir::Operation* createBitCastOp(mlir::Type resultType, mlir::Value input) {
        auto op = builder_->create<mlir::tensor::BitcastOp>(
            builder_->getUnknownLoc(), resultType, input);
		opStack_->push(op.getOperation());
        return op.getOperation();
	}

    mlir::Operation* createCastOp(mlir::Type resultType, mlir::Value input) {
        auto op = builder_->create<mlir::tensor::CastOp>(
            builder_->getUnknownLoc(), resultType, input);
        opStack_->push(op.getOperation());
        return op.getOperation();
	}

    mlir::Operation* createCollapseShapeOp(mlir::Type resultType, mlir::Value input, mlir::ArrayRef<mlir::ReassociationIndices> reassociation) {
        auto op = builder_->create<mlir::tensor::CollapseShapeOp>(
            builder_->getUnknownLoc(), resultType, input, reassociation);
        opStack_->push(op.getOperation());
        return op.getOperation();
    }

    mlir::Operation* createConcatOp(mlir::Type resultType, mlir::ValueRange inputs, int64_t dimension) {
        // Use the overload that takes (loc, resultType, dim, inputs)
        auto op = builder_->create<mlir::tensor::ConcatOp>(
            builder_->getUnknownLoc(), resultType, dimension, inputs);
        opStack_->push(op.getOperation());
        return op.getOperation();
	}

    mlir::Operation* createDimOp(mlir::Value source, int64_t index) {
        auto op = builder_->create<mlir::tensor::DimOp>(builder_->getUnknownLoc(), source, index);
        opStack_->push(op.getOperation());
        return op.getOperation();
    }

    mlir::Operation* createEmptyOp(mlir::Type elementType, llvm::ArrayRef<int64_t> staticShape) {
        auto op = builder_->create<mlir::tensor::EmptyOp>(builder_->getUnknownLoc(), staticShape, elementType);
        opStack_->push(op.getOperation());
        return op.getOperation();
    }

    mlir::Operation* createExpandShapeOp(mlir::Type resultType, mlir::Value input, mlir::ArrayRef<mlir::ReassociationIndices> reassociation) {
        auto op = builder_->create<mlir::tensor::ExpandShapeOp>(builder_->getUnknownLoc(), resultType, input, reassociation);
        opStack_->push(op.getOperation());
        return op.getOperation();
    }

    mlir::Operation* createExtractOp(mlir::Type resultType, mlir::Value input, mlir::ValueRange indices) {
        auto op = builder_->create<mlir::tensor::ExtractOp>(builder_->getUnknownLoc(), resultType, input, indices);
        opStack_->push(op.getOperation());
        return op.getOperation();
    }

    mlir::Operation* createExtractSliceOp(mlir::Value input, mlir::ArrayAttr offsets,
                                        mlir::ArrayAttr sizes, mlir::ArrayAttr strides) {
        auto off = toOpFoldResults(offsets);
        auto sz = toOpFoldResults(sizes);
        auto st = toOpFoldResults(strides);
        auto op = builder_->create<mlir::tensor::ExtractSliceOp>(builder_->getUnknownLoc(), input,
                                                                  off, sz, st);
        opStack_->push(op.getOperation());
        return op.getOperation();
    }

    mlir::Operation* createFromElementsOp(mlir::Type resultType, mlir::ValueRange elements) {
        auto op = builder_->create<mlir::tensor::FromElementsOp>(builder_->getUnknownLoc(), resultType, elements);
        opStack_->push(op.getOperation());
        return op.getOperation();
    }

    mlir::Operation* createGatherOp(mlir::Type resultType, mlir::Value source, mlir::Value indices, llvm::ArrayRef<int64_t> gatherDims = {}) {
        auto op = builder_->create<mlir::tensor::GatherOp>(builder_->getUnknownLoc(), resultType, source, indices, gatherDims);
        opStack_->push(op.getOperation());
        return op.getOperation();
    }

    mlir::Operation* createGenerateOp(mlir::Type resultType, mlir::ValueRange dynamicExtents = {}) {
        auto op = builder_->create<mlir::tensor::GenerateOp>(builder_->getUnknownLoc(), resultType, dynamicExtents);
        opStack_->push(op.getOperation());
        return op.getOperation();
    }

    mlir::Operation* createInsertOp(mlir::Value scalar, mlir::Value dest, mlir::ValueRange indices) {
        auto op = builder_->create<mlir::tensor::InsertOp>(builder_->getUnknownLoc(), scalar, dest, indices);
        opStack_->push(op.getOperation());
        return op.getOperation();
    }

    mlir::Operation* createInsertSliceOp(mlir::Value source, mlir::Value dest, mlir::ArrayAttr offsets,
                                    mlir::ArrayAttr sizes, mlir::ArrayAttr strides) {
        auto off = toOpFoldResults(offsets);
        auto sz = toOpFoldResults(sizes);
        auto st = toOpFoldResults(strides);
        auto op = builder_->create<mlir::tensor::InsertSliceOp>(builder_->getUnknownLoc(), source, dest, off, sz, st);
        opStack_->push(op.getOperation());
        return op.getOperation();
    }

    mlir::Operation* createPadOp(mlir::Type resultType, mlir::Value input,  mlir::ValueRange lowPad, mlir::ValueRange highPad) {
        auto op = builder_->create<mlir::tensor::PadOp>(builder_->getUnknownLoc(), resultType, input, lowPad, highPad);
        opStack_->push(op.getOperation());
        return op.getOperation();
    }

    mlir::Operation* createParallelInsertSliceOp(mlir::Value source, mlir::Value dest, mlir::ArrayAttr offsets,
                                            mlir::ArrayAttr sizes, mlir::ArrayAttr strides) {
        auto off = toOpFoldResults(offsets);
        auto sz = toOpFoldResults(sizes);
        auto st = toOpFoldResults(strides);
        auto op = builder_->create<mlir::tensor::ParallelInsertSliceOp>(builder_->getUnknownLoc(), source, dest, off, sz, st);
        opStack_->push(op.getOperation());
        return op.getOperation();
    }

    mlir::Operation* createRankOp(mlir::Value input) {
        auto op = builder_->create<mlir::tensor::RankOp>(builder_->getUnknownLoc(), input);
        opStack_->push(op.getOperation());
        return op.getOperation();
    }

    mlir::Operation* createReshapeOp(mlir::Value source, mlir::Value shape, mlir::Type resultType) {
        auto op = builder_->create<mlir::tensor::ReshapeOp>(builder_->getUnknownLoc(), resultType, source, shape);
        opStack_->push(op.getOperation());
        return op.getOperation();
    }

    mlir::Operation* createScatterOp(mlir::Type resultType, mlir::Value source, mlir::Value dest, mlir::Value indices, mlir::ArrayRef<int64_t> scatterDims) {
        auto op = builder_->create<mlir::tensor::ScatterOp>(builder_->getUnknownLoc(), resultType, source, dest, indices, scatterDims);
        opStack_->push(op.getOperation());
        return op.getOperation();
    }

    mlir::Operation* createSplatOp(mlir::Type resultType, mlir::Value element, mlir::ValueRange dynamicSizes = {}) {
        auto op = builder_->create<mlir::tensor::SplatOp>(builder_->getUnknownLoc(), element, resultType, dynamicSizes);
        opStack_->push(op.getOperation());
        return op.getOperation();
    }

    mlir::Operation* createYieldOp(mlir::Value result) {
        auto op = builder_->create<mlir::tensor::YieldOp>(builder_->getUnknownLoc(), result);     
        opStack_->push(op.getOperation());
        return op.getOperation();
    }


private:
    // Helpers
    mlir::ArrayRef<mlir::OpFoldResult> toOpFoldResults(mlir::ArrayAttr attr) {
        llvm::SmallVector<mlir::OpFoldResult> res;
        if (!attr)
            return res;
        for (auto a : attr.getValue())
            res.push_back(mlir::OpFoldResult(a));
        return mlir::ArrayRef<mlir::OpFoldResult>(res);
    }

	

};


class TosaOps {
private:
    std::shared_ptr<mlir::OpBuilder> builder_;
    std::shared_ptr<std::stack<Compiler::StackEntry>> opStack_;


public:
    TosaOps(const Compiler& compiler) {
        builder_ = std::make_shared<mlir::OpBuilder>(compiler.getContext());
        opStack_ = compiler.getOpStack();
        module_ = compiler.getModule();
        context_ = compiler.getContext();
        funcCounter_ = 0;
    }

    // Helper to create a new function for each op
    mlir::func::FuncOp createFuncOp(mlir::Type resultType, llvm::ArrayRef<mlir::Type> argTypes = {}, llvm::ArrayRef<mlir::Value> args = {}) {
        std::string funcName = "tosa_func_" + std::to_string(funcCounter_++);
        auto funcType = builder_->getFunctionType(argTypes, {resultType});
        auto funcOp = builder_->create<mlir::func::FuncOp>(builder_->getUnknownLoc(), funcName, funcType);
        funcOp.addEntryBlock();
        module_.push_back(funcOp);
        auto callOp = builder_->create<mlir::func::CallOp>(builder_->getUnknownLoc(), funcName, resultType, args);
        opStack_->push(callOp.getOperation());
        return funcOp;
    }

    // TOSA op creators: each op in its own function (FuncOp)
    mlir::Operation* createAbsOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()}, {input});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::AbsOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createAddOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
        auto funcOp = createFuncOp(resultType, {lhs.getType(), rhs.getType()}, {lhs, rhs});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::AddOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());        
        return op.getOperation();
    }

    mlir::Operation* createApplyScaleOp(mlir::Type resultType, mlir::Value input, mlir::Value scale, mlir::Value shift, bool inputRounding) {
        auto funcOp = createFuncOp(resultType, {input.getType(), scale.getType(), shift.getType()}, {input, scale, shift});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ApplyScaleOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1), funcOp.getArgument(2), inputRounding);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());        
        return op.getOperation();
    }

    mlir::Operation* createArgMaxOp(mlir::Type resultType, mlir::Value input, int32_t axis) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ArgMaxOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), axis);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());        
        return op.getOperation();
    }

    mlir::Operation* createArithmeticRightShiftOp(mlir::Type resultType, mlir::Value input, mlir::Value input2, bool round) {
        auto funcOp = createFuncOp(resultType, {input.getType(), input2.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ArithmeticRightShiftOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1), round);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());        
        return op.getOperation();
    }

    mlir::Operation* createAvgPool2dOp(mlir::Type resultType, mlir::Value input, mlir::ArrayRef<int64_t> filterHeightWidth, mlir::ArrayRef<int64_t> strides, mlir::ArrayRef<int64_t> padding, mlir::Type accumlation_type) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::AvgPool2dOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), filterHeightWidth, strides, padding, accumlation_type, nullptr);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createBitwiseAndOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
        auto funcOp = createFuncOp(resultType, {lhs.getType(), rhs.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::BitwiseAndOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createBitwiseNotOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::BitwiseNotOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createBitwiseOrOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
        auto funcOp = createFuncOp(resultType, {lhs.getType(), rhs.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::BitwiseOrOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createBitwiseXorOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
        auto funcOp = createFuncOp(resultType, {lhs.getType(), rhs.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::BitwiseXorOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createCastOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::CastOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createCeilOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::CeilOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createClampOp(mlir::Type resultType, mlir::Value input, uint64_t min, uint64_t max, float min_fp, float max_fp) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto minFpAttr = builder_->getF32FloatAttr(min_fp);
        auto maxFpAttr = builder_->getF32FloatAttr(max_fp);
        auto op = builder_->create<mlir::tosa::ClampOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), min, max, minFpAttr, maxFpAttr);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createClzOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ClzOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createConcatOp(mlir::Type resultType, mlir::ValueRange inputs, int32_t axis) {
        // ValueRange is not a type, so we need to handle this differently. We'll assume inputs is a vector of values.
        std::vector<mlir::Type> argTypes;
        for (auto v : inputs) argTypes.push_back(v.getType());
        auto funcOp = createFuncOp(resultType, argTypes);
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        std::vector<mlir::Value> funcArgs;
        for (size_t i = 0; i < argTypes.size(); ++i) funcArgs.push_back(funcOp.getArgument(i));
        auto op = builder_->create<mlir::tosa::ConcatOp>(builder_->getUnknownLoc(), resultType, funcArgs, axis);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createConstOp(mlir::Type resultType, mlir::ElementsAttr value) {
        auto funcOp = createFuncOp(resultType, {});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ConstOp>(builder_->getUnknownLoc(), resultType, value);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createConstShapeOp(mlir::Type resultType, mlir::DenseIntElementsAttr value) {
        auto funcOp = createFuncOp(resultType, {});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ConstShapeOp>(builder_->getUnknownLoc(), resultType, value);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createConv2DOp(mlir::Type resultType, mlir::Value input, mlir::Value filter, mlir::Value bias, llvm::ArrayRef<int64_t> strides, llvm::ArrayRef<int64_t> dilations, llvm::ArrayRef<int64_t> paddings, mlir::Type accType) {
        auto funcOp = createFuncOp(resultType, {input.getType(), filter.getType(), bias.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::Conv2DOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1), funcOp.getArgument(2), paddings, strides, dilations, accType, nullptr, false);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createConv3DOp(mlir::Type resultType, mlir::Value input, mlir::Value filter, mlir::Value bias, llvm::ArrayRef<int64_t> strides, llvm::ArrayRef<int64_t> dilations, llvm::ArrayRef<int64_t> paddings, mlir::Type accType) {
        auto funcOp = createFuncOp(resultType, {input.getType(), filter.getType(), bias.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::Conv3DOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1), funcOp.getArgument(2), paddings, strides, dilations, accType, nullptr, false);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createCosOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::CosOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createCustomOp(mlir::Type resultType, mlir::StringAttr customCode, mlir::ValueRange inputs, llvm::ArrayRef<int64_t> customShape = {}) {
        // Not implemented: custom op signature is not standard.
        return nullptr;
    }

    mlir::Operation* createDepthwiseConv2DOp(mlir::Type resultType, mlir::Value input, mlir::Value filter, mlir::Value bias, llvm::ArrayRef<int64_t> strides, llvm::ArrayRef<int64_t> dilations, llvm::ArrayRef<int64_t> paddings, mlir::Type accType) {
        auto funcOp = createFuncOp(resultType, {input.getType(), filter.getType(), bias.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::DepthwiseConv2DOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1), funcOp.getArgument(2), paddings, strides, dilations, accType, nullptr, false);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createEqualOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
        auto funcOp = createFuncOp(resultType, {lhs.getType(), rhs.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::EqualOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());        
        return op.getOperation();
    }

    mlir::Operation* createErfOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ErfOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createExpOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ExpOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createFFT2dOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::FFT2dOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult(0));
        
        return op.getOperation();
    }

    mlir::Operation* createFloorOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::FloorOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createGatherOp(mlir::Type resultType, mlir::Value input, mlir::Value indices) {
        auto funcOp = createFuncOp(resultType, {input.getType(), indices.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::GatherOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createGreaterEqualOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
        auto funcOp = createFuncOp(resultType, {lhs.getType(), rhs.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::GreaterEqualOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createGreaterOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
        auto funcOp = createFuncOp(resultType, {lhs.getType(), rhs.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::GreaterOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createIdentityOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::IdentityOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createIfOp(mlir::Type resultType, mlir::Value cond, mlir::Region& thenRegion, mlir::Region& elseRegion) {
        // Not implemented: region-based ops need special handling.
        return nullptr;
    }

    mlir::Operation* createIntDivOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
        auto funcOp = createFuncOp(resultType, {lhs.getType(), rhs.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::IntDivOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createLogOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::LogOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createLogicalAndOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
        auto funcOp = createFuncOp(resultType, {lhs.getType(), rhs.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::LogicalAndOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createLogicalLeftShiftOp(mlir::Type resultType, mlir::Value input, mlir::Value shift) {
        auto funcOp = createFuncOp(resultType, {input.getType(), shift.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::LogicalLeftShiftOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createLogicalNotOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::LogicalNotOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createLogicalOrOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
        auto funcOp = createFuncOp(resultType, {lhs.getType(), rhs.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::LogicalOrOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createLogicalRightShiftOp(mlir::Type resultType, mlir::Value input, mlir::Value shift) {
        auto funcOp = createFuncOp(resultType, {input.getType(), shift.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::LogicalRightShiftOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createLogicalXorOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
        auto funcOp = createFuncOp(resultType, {lhs.getType(), rhs.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::LogicalXorOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createMatMulOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
        auto funcOp = createFuncOp(resultType, {lhs.getType(), rhs.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::MatMulOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createMaxPool2dOp(mlir::Type resultType, mlir::Value input, llvm::ArrayRef<int64_t> filterHeightWidth, llvm::ArrayRef<int64_t> strides, llvm::ArrayRef<int64_t> paddings) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::MaxPool2dOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), filterHeightWidth, strides, paddings);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createMaximumOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
        auto funcOp = createFuncOp(resultType, {lhs.getType(), rhs.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::MaximumOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createMinimumOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
        auto funcOp = createFuncOp(resultType, {lhs.getType(), rhs.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::MinimumOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createMulOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs, uint8_t shift) {
        auto funcOp = createFuncOp(resultType, {lhs.getType(), rhs.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::MulOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1), shift);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createNegateOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::NegateOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createPadOp(mlir::Type resultType, mlir::Value input, mlir::Value paddings, mlir::Value padValue = 0) {
        auto funcOp = createFuncOp(resultType, {input.getType(), paddings.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::PadOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1), padValue);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createPowOp(mlir::Type resultType, mlir::Value input, mlir::Value exponent) {
        auto funcOp = createFuncOp(resultType, {input.getType(), exponent.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::PowOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createRFFT2dOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::RFFT2dOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult(0));
        
        return op.getOperation();
    }

    mlir::Operation* createReciprocalOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ReciprocalOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createReduceAllOp(mlir::Type resultType, mlir::Value input, mlir::IntegerAttr axes) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ReduceAllOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), axes);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createReduceAnyOp(mlir::Type resultType, mlir::Value input, mlir::IntegerAttr axes) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ReduceAnyOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), axes);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createReduceMaxOp(mlir::Type resultType, mlir::Value input, mlir::IntegerAttr axes) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ReduceMaxOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), axes);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createReduceMinOp(mlir::Type resultType, mlir::Value input, mlir::IntegerAttr axes) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ReduceMinOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), axes);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createReduceProductOp(mlir::Type resultType, mlir::Value input, mlir::IntegerAttr axes) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ReduceProdOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), axes);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createReduceSumOp(mlir::Type resultType, mlir::Value input, mlir::IntegerAttr axes) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ReduceSumOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), axes);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createRescaleOp(mlir::Type resultType, mlir::Value input, uint32_t input_zp, uint32_t output_zp, mlir::ArrayRef<int32_t> multiplier, mlir::ArrayRef<int8_t> shift, bool scale32, bool double_round, bool per_channel, bool input_unsigned=false, bool output_unsigned=false) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::RescaleOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), input_zp, output_zp, multiplier, shift, scale32, double_round, per_channel, input_unsigned, output_unsigned);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createReshapeOp(mlir::Type resultType, mlir::Value input, mlir::ArrayRef<int64_t> shape) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ReshapeOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), shape);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createResizeOp(mlir::Type resultType, mlir::Value input, mlir::ArrayRef<int64_t> newSize, mlir::ArrayRef<int64_t> offset, mlir::ArrayRef<int64_t> border, mlir::StringRef mode) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ResizeOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), newSize, offset, border, mode);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createReverseOp(mlir::Type resultType, mlir::Value input, uint32_t axes) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ReverseOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), axes);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createRsqrtOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::RsqrtOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createScatterOp(mlir::Type resultType, mlir::Value input, mlir::Value indices, mlir::Value updates) {
        auto funcOp = createFuncOp(resultType, {input.getType(), indices.getType(), updates.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::ScatterOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(2), funcOp.getArgument(1), funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createSelectOp(mlir::Type resultType, mlir::Value cond, mlir::Value onTrue, mlir::Value onFalse) {
        auto funcOp = createFuncOp(resultType, {cond.getType(), onTrue.getType(), onFalse.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::SelectOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1), funcOp.getArgument(2));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createSigmoidOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::SigmoidOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createSinOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::SinOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createSliceOp(mlir::Type resultType, mlir::Value input, llvm::ArrayRef<int64_t> starts, llvm::ArrayRef<int64_t> sizes) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::SliceOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), starts, sizes);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createSubOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
        auto funcOp = createFuncOp(resultType, {lhs.getType(), rhs.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::SubOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createTableOp(mlir::Type resultType, mlir::Value input, mlir::Value table) {
        auto funcOp = createFuncOp(resultType, {input.getType(), table.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::TableOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createTanhOp(mlir::Type resultType, mlir::Value input) {
        auto funcOp = createFuncOp(resultType, {input.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::TanhOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createTileOp(mlir::Type resultType, mlir::Value input, mlir::Value multiples) {
        auto funcOp = createFuncOp(resultType, {input.getType(), multiples.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::TileOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createTransposeConv2DOp(mlir::Type resultType, mlir::Value input, mlir::Value filter, mlir::Value bias, llvm::ArrayRef<int64_t> outputShape, llvm::ArrayRef<int64_t> strides, llvm::ArrayRef<int64_t> paddings, mlir::Type accType) {
        auto funcOp = createFuncOp(resultType, {input.getType(), filter.getType(), bias.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::TransposeConv2DOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1), funcOp.getArgument(2), paddings, strides, outputShape, accType, nullptr, false);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createTransposeOp(mlir::Type resultType, mlir::Value input, mlir::Value perm) {
        auto funcOp = createFuncOp(resultType, {input.getType(), perm.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::TransposeOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0), funcOp.getArgument(1));
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult());
        
        return op.getOperation();
    }

    mlir::Operation* createVariableOp(mlir::Type resultType, mlir::StringAttr name, mlir::Attribute initial_value) {
        auto funcOp = createFuncOp(resultType, {});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::VariableOp>(builder_->getUnknownLoc(), name, resultType, initial_value);
        //builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResults(0));
        
        return op.getOperation();
    }

    mlir::Operation* createVariableReadOp(mlir::Type resultType, mlir::Value variable) {
        auto funcOp = createFuncOp(resultType, {variable.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::VariableReadOp>(builder_->getUnknownLoc(), resultType, funcOp.getArgument(0));
        //builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResults(0));
        
        return op.getOperation();
    }

    mlir::Operation* createVariableWriteOp(mlir::StringRef name, mlir::Value value) {
        auto funcOp = createFuncOp(value.getType(), {value.getType()});
        builder_->setInsertionPointToStart(&funcOp.getBody().front());
        auto op = builder_->create<mlir::tosa::VariableWriteOp>(builder_->getUnknownLoc(), name, funcOp.getArgument(0));
        //builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResults(0));
        
        return op.getOperation();
    }

    mlir::Operation* createWhileOp(mlir::TypeRange resultType, mlir::ValueRange operands) {
        // Not implemented: region-based ops need special handling.
        return nullptr;
    }

    mlir::Operation* createYieldOp(mlir::ValueRange results) {
        // Not implemented: region-based ops need special handling.
        return nullptr;
    }

    // ...existing code...

private:
    mlir::ModuleOp module_;
    mlir::MLIRContext* context_;
    int funcCounter_;

    /*
    Operation definitions
    tosa.abs (mlir::tosa::AbsOp)
    tosa.add (mlir::tosa::AddOp)
    tosa.apply_scale (mlir::tosa::ApplyScaleOp)
    tosa.argmax (mlir::tosa::ArgMaxOp)
    tosa.arithmetic_right_shift (mlir::tosa::ArithmeticRightShiftOp)
    tosa.avg_pool2d (mlir::tosa::AvgPool2dOp)
    tosa.bitwise_and (mlir::tosa::BitwiseAndOp)
    tosa.bitwise_not (mlir::tosa::BitwiseNotOp)
    tosa.bitwise_or (mlir::tosa::BitwiseOrOp)
    tosa.bitwise_xor (mlir::tosa::BitwiseXorOp)
    tosa.cast (mlir::tosa::CastOp)
    tosa.ceil (mlir::tosa::CeilOp)
    tosa.clamp (mlir::tosa::ClampOp)
    tosa.clz (mlir::tosa::ClzOp)
    tosa.concat (mlir::tosa::ConcatOp)
    tosa.const (mlir::tosa::ConstOp)
    tosa.const_shape (mlir::tosa::ConstShapeOp)
    tosa.conv2d (mlir::tosa::Conv2DOp)
    tosa.conv3d (mlir::tosa::Conv3DOp)
    tosa.cos (mlir::tosa::CosOp)
    tosa.custom (mlir::tosa::CustomOp)
    tosa.depthwise_conv2d (mlir::tosa::DepthwiseConv2DOp)
    tosa.equal (mlir::tosa::EqualOp)
    tosa.erf (mlir::tosa::ErfOp)
    tosa.exp (mlir::tosa::ExpOp)
    tosa.fft2d (mlir::tosa::FFT2dOp)
    tosa.floor (mlir::tosa::FloorOp)
    tosa.gather (mlir::tosa::GatherOp)
    tosa.greater_equal (mlir::tosa::GreaterEqualOp)
    tosa.greater (mlir::tosa::GreaterOp)
    tosa.identity (mlir::tosa::IdentityOp)
    tosa.cond_if (mlir::tosa::IfOp)
    tosa.intdiv (mlir::tosa::IntDivOp)
    tosa.log (mlir::tosa::LogOp)
    tosa.logical_and (mlir::tosa::LogicalAndOp)
    tosa.logical_left_shift (mlir::tosa::LogicalLeftShiftOp)
    tosa.logical_not (mlir::tosa::LogicalNotOp)
    tosa.logical_or (mlir::tosa::LogicalOrOp)
    tosa.logical_right_shift (mlir::tosa::LogicalRightShiftOp)
    tosa.logical_xor (mlir::tosa::LogicalXorOp)
    tosa.matmul (mlir::tosa::MatMulOp)
    tosa.max_pool2d (mlir::tosa::MaxPool2dOp)
    tosa.maximum (mlir::tosa::MaximumOp)
    tosa.minimum (mlir::tosa::MinimumOp)
    tosa.mul (mlir::tosa::MulOp)
    tosa.negate (mlir::tosa::NegateOp)
    tosa.pad (mlir::tosa::PadOp)
    tosa.pow (mlir::tosa::PowOp)
    tosa.rfft2d (mlir::tosa::RFFT2dOp)
    tosa.reciprocal (mlir::tosa::ReciprocalOp)
    tosa.reduce_all (mlir::tosa::ReduceAllOp)
    tosa.reduce_any (mlir::tosa::ReduceAnyOp)
    tosa.reduce_max (mlir::tosa::ReduceMaxOp)
    tosa.reduce_min (mlir::tosa::ReduceMinOp)
    tosa.reduce_product (mlir::tosa::ReduceProductOp)
    tosa.reduce_sum (mlir::tosa::ReduceSumOp)
    tosa.rescale (mlir::tosa::RescaleOp)
    tosa.reshape (mlir::tosa::ReshapeOp)
    tosa.resize (mlir::tosa::ResizeOp)
    tosa.reverse (mlir::tosa::ReverseOp)
    tosa.rsqrt (mlir::tosa::RsqrtOp)
    tosa.scatter (mlir::tosa::ScatterOp)
    tosa.select (mlir::tosa::SelectOp)
    tosa.sigmoid (mlir::tosa::SigmoidOp)
    tosa.sin (mlir::tosa::SinOp)
    tosa.slice (mlir::tosa::SliceOp)
    tosa.sub (mlir::tosa::SubOp)
    tosa.table (mlir::tosa::TableOp)
    tosa.tanh (mlir::tosa::TanhOp)
    tosa.tile (mlir::tosa::TileOp)
    tosa.transpose_conv2d (mlir::tosa::TransposeConv2DOp)
    tosa.transpose (mlir::tosa::TransposeOp)
    tosa.variable (mlir::tosa::VariableOp)
    tosa.variable_read (mlir::tosa::VariableReadOp)
    tosa.variable_write (mlir::tosa::VariableWriteOp)
    tosa.while_loop (mlir::tosa::WhileOp)
    tosa.yield (mlir::tosa::YieldOp)
    */


//     // Macro to generate a function for each Tosa op
// #define CREATE_TOSA_OP_FUNC(OPNAME, ...) \
//     mlir::Operation* create##OPNAME##Op(__VA_ARGS__) { \
//         std::vector<mlir::Type> argTypes; \
//         std::vector<mlir::Value> argVals; \
//         int _idx = 0; \
//         (void)_idx; \
//         /* Collect argument types and values */ \
//         auto collectArg = [&](auto&& arg) { \
//             using T = std::decay_t<decltype(arg)>; \
//             if constexpr (std::is_base_of_v<mlir::Value, T>) { \
//                 argTypes.push_back(arg.getType()); \
//                 argVals.push_back(arg); \
//             } \
//         }; \
//         (collectArg(__VA_ARGS__), ...); \
//         auto funcOp = createFuncOp(resultType, argTypes); \
//         builder_->setInsertionPointToStart(&funcOp.getBody().front()); \
//         std::vector<mlir::Value> funcArgs; \
//         for (size_t i = 0; i < argTypes.size(); ++i) funcArgs.push_back(funcOp.getArgument(i)); \
//         auto op = builder_->create<mlir::tosa::OPNAME##Op>(builder_->getUnknownLoc(), resultType, funcArgs); \
//         builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc(), op.getResult()); \
//         opStack_->push(op.getOperation()); \
//         return op.getOperation(); \
//     }

//     // Use the macro for all Tosa ops (add more as needed)
//     CREATE_TOSA_OP_FUNC(Abs, mlir::Type resultType, mlir::Value input)
//     CREATE_TOSA_OP_FUNC(Add, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
//     CREATE_TOSA_OP_FUNC(ApplyScale, mlir::Type resultType, mlir::Value input, mlir::Value scale, mlir::Value shift, bool inputRounding)
//     CREATE_TOSA_OP_FUNC(ArgMax, mlir::Type resultType, mlir::Value input, int32_t axis)
//     CREATE_TOSA_OP_FUNC(ArithmeticRightShift, mlir::Type resultType, mlir::Value input, mlir::Value input2, bool round)
//     CREATE_TOSA_OP_FUNC(AvgPool2d, mlir::Type resultType, mlir::Value input, mlir::ArrayRef<int64_t> filterHeightWidth, mlir::ArrayRef<int64_t> strides, mlir::ArrayRef<int64_t> padding, mlir::Type accumlation_type)
//     CREATE_TOSA_OP_FUNC(BitwiseAnd, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
//     CREATE_TOSA_OP_FUNC(BitwiseNot, mlir::Type resultType, mlir::Value input)
//     CREATE_TOSA_OP_FUNC(BitwiseOr, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
//     CREATE_TOSA_OP_FUNC(BitwiseXor, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
//     CREATE_TOSA_OP_FUNC(Cast, mlir::Type resultType, mlir::Value input)
//     CREATE_TOSA_OP_FUNC(Ceil, mlir::Type resultType, mlir::Value input)
//     CREATE_TOSA_OP_FUNC(Clamp, mlir::Type resultType, mlir::Value input, uint64_t min, uint64_t max, float min_fp, float max_fp)
//     CREATE_TOSA_OP_FUNC(Clz, mlir::Type resultType, mlir::Value input)
//     CREATE_TOSA_OP_FUNC(Concat, mlir::Type resultType, mlir::ValueRange inputs, int32_t axis)
//     CREATE_TOSA_OP_FUNC(Const, mlir::Type resultType, mlir::ElementsAttr value)
//     CREATE_TOSA_OP_FUNC(ConstShape, mlir::Type resultType, mlir::DenseIntElementsAttr value)
//     CREATE_TOSA_OP_FUNC(Conv2D, mlir::Type resultType, mlir::Value input, mlir::Value filter, mlir::Value bias, llvm::ArrayRef<int64_t> strides, llvm::ArrayRef<int64_t> dilations, llvm::ArrayRef<int64_t> paddings, mlir::Type accType)
//     CREATE_TOSA_OP_FUNC(Conv3D, mlir::Type resultType, mlir::Value input, mlir::Value filter, mlir::Value bias, llvm::ArrayRef<int64_t> strides, llvm::ArrayRef<int64_t> dilations, llvm::ArrayRef<int64_t> paddings, mlir::Type accType)
//     CREATE_TOSA_OP_FUNC(Cos, mlir::Type resultType, mlir::Value input)
//     CREATE_TOSA_OP_FUNC(DepthwiseConv2D, mlir::Type resultType, mlir::Value input, mlir::Value filter, mlir::Value bias, llvm::ArrayRef<int64_t> strides, llvm::ArrayRef<int64_t> dilations, llvm::ArrayRef<int64_t> paddings, mlir::Type accType)
//     CREATE_TOSA_OP_FUNC(Equal, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs)
//     CREATE_TOSA_OP_FUNC(Erf, mlir::Type resultType, mlir::Value input)
//     CREATE_TOSA_OP_FUNC(Exp, mlir::Type resultType, mlir::Value input)
//     CREATE_TOSA_OP_FUNC(FFT2d, mlir::Type resultType, mlir::Value input)
//     CREATE_TOSA_OP_FUNC(Floor, mlir::Type resultType, mlir::Value input)
//     // ...repeat for all other Tosa ops as needed...

// #undef CREATE_TOSA_OP_FUNC

//     mlir::Operation* createGatherOp(mlir::Type resultType, mlir::Value input, mlir::Value indices) {
//         auto op = builder_->create<mlir::tosa::GatherOp>(builder_->getUnknownLoc(), resultType, input, indices);
//         return op.getOperation();
// 	}

//     mlir::Operation* createGreaterEqualOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
//         auto op = builder_->create<mlir::tosa::GreaterEqualOp>(builder_->getUnknownLoc(), resultType, lhs, rhs);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
// 	}

//     mlir::Operation* createGreaterOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
//         auto op = builder_->create<mlir::tosa::GreaterOp>(builder_->getUnknownLoc(), resultType, lhs, rhs);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
// 	}

//     mlir::Operation* createIdentityOp(mlir::Type resultType, mlir::Value input) {
//         auto op = builder_->create<mlir::tosa::IdentityOp>(builder_->getUnknownLoc(), resultType, input);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
// 	}

//     mlir::Operation* createIfOp(mlir::Type resultType, mlir::Value cond, mlir::Region& thenRegion, mlir::Region& elseRegion) {
//         // auto op = builder_->create<mlir::tosa::IfOp>(builder_->getUnknownLoc(), resultType, cond, thenRegion, elseRegion);
//         //return op.getOperation(); // Assuming single result
//         return nullptr;
//     }

//     mlir::Operation* createIntDivOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
//         auto op = builder_->create<mlir::tosa::IntDivOp>(builder_->getUnknownLoc(), resultType, lhs, rhs);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createLogOp(mlir::Type resultType, mlir::Value input) {
//         auto op = builder_->create<mlir::tosa::LogOp>(builder_->getUnknownLoc(), resultType, input);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createLogicalAndOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
//         auto op = builder_->create<mlir::tosa::LogicalAndOp>(builder_->getUnknownLoc(), resultType, lhs, rhs);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createLogicalLeftShiftOp(mlir::Type resultType, mlir::Value input, mlir::Value shift) {
//         auto op = builder_->create<mlir::tosa::LogicalLeftShiftOp>(builder_->getUnknownLoc(), resultType, input, shift);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createLogicalNotOp(mlir::Type resultType, mlir::Value input) {
//         auto op = builder_->create<mlir::tosa::LogicalNotOp>(builder_->getUnknownLoc(), resultType, input);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createLogicalOrOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
//         auto op = builder_->create<mlir::tosa::LogicalOrOp>(builder_->getUnknownLoc(), resultType, lhs, rhs);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createLogicalRightShiftOp(mlir::Type resultType, mlir::Value input, mlir::Value shift) {
//         auto op = builder_->create<mlir::tosa::LogicalRightShiftOp>(builder_->getUnknownLoc(), resultType, input, shift);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createLogicalXorOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
//         auto op = builder_->create<mlir::tosa::LogicalXorOp>(builder_->getUnknownLoc(), resultType, lhs, rhs);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
// 	}

//     mlir::Operation* createMatMulOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
//         auto op = builder_->create<mlir::tosa::MatMulOp>(builder_->getUnknownLoc(), resultType, lhs, rhs);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createMaxPool2dOp(mlir::Type resultType, mlir::Value input, llvm::ArrayRef<int64_t> filterHeightWidth,
//                                   llvm::ArrayRef<int64_t> strides, llvm::ArrayRef<int64_t> paddings) {
//         auto op = builder_->create<mlir::tosa::MaxPool2dOp>(builder_->getUnknownLoc(), resultType, input,
//             filterHeightWidth, strides, paddings);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createMaximumOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs){
//         auto op = builder_->create<mlir::tosa::MaximumOp>(builder_->getUnknownLoc(), resultType, lhs, rhs);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createMinimumOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
//         auto op = builder_->create<mlir::tosa::MinimumOp>(builder_->getUnknownLoc(), resultType, lhs, rhs);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createMulOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs, uint8_t shift) {
// 		auto op = builder_->create <mlir::tosa::MulOp>(builder_->getUnknownLoc(), resultType, lhs, rhs, shift);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createNegateOp(mlir::Type resultType, mlir::Value input) {
//         auto op = builder_->create<mlir::tosa::NegateOp>(builder_->getUnknownLoc(), resultType, input);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createPadOp(mlir::Type resultType, mlir::Value input, mlir::Value paddings, mlir::Value padValue = 0) {
//         auto op = builder_->create<mlir::tosa::PadOp>(builder_->getUnknownLoc(), resultType, input, paddings, padValue);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createPowOp(mlir::Type resultType, mlir::Value input, mlir::Value exponent) {
//         auto op = builder_->create<mlir::tosa::PowOp>(builder_->getUnknownLoc(), resultType, input, exponent);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createRFFT2dOp(mlir::Type resultType, mlir::Value input) {
//         auto op = builder_->create<mlir::tosa::RFFT2dOp>(builder_->getUnknownLoc(), resultType, input);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createReciprocalOp(mlir::Type resultType, mlir::Value input) {
//         auto op = builder_->create<mlir::tosa::ReciprocalOp>(builder_->getUnknownLoc(), resultType, input);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createReduceAllOp(mlir::Type resultType, mlir::Value input, mlir::IntegerAttr axes) {
//         auto op = builder_->create<mlir::tosa::ReduceAllOp>(builder_->getUnknownLoc(), resultType, input, axes);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

// 	mlir::Operation* createReduceAnyOp(mlir::Type resultType, mlir::Value input, mlir::IntegerAttr axes) {
//         auto op = builder_->create<mlir::tosa::ReduceAnyOp>(builder_->getUnknownLoc(), resultType, input, axes);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createReduceMaxOp(mlir::Type resultType, mlir::Value input, mlir::IntegerAttr axes) {
//         auto op = builder_->create<mlir::tosa::ReduceMaxOp>(builder_->getUnknownLoc(), resultType, input, axes);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createReduceMinOp(mlir::Type resultType, mlir::Value input, mlir::IntegerAttr axes) {
//         auto op = builder_->create<mlir::tosa::ReduceMinOp>(builder_->getUnknownLoc(), resultType, input, axes);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createReduceProductOp(mlir::Type resultType, mlir::Value input, mlir::IntegerAttr axes) {
//         auto op = builder_->create<mlir::tosa::ReduceProdOp>(builder_->getUnknownLoc(), resultType, input, axes);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createReduceSumOp(mlir::Type resultType, mlir::Value input, mlir::IntegerAttr axes) {
//         auto op = builder_->create<mlir::tosa::ReduceSumOp>(builder_->getUnknownLoc(), resultType, input, axes);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

// 	mlir::Operation* createRescaleOp(mlir::Type resultType, mlir::Value input, uint32_t input_zp, uint32_t output_zp, mlir::ArrayRef<int32_t> multiplier, mlir::ArrayRef<int8_t> shift, bool scale32, bool double_round, bool per_channel, bool input_unsigned=false, bool output_unsigned=false) {
//         auto op = builder_->create<mlir::tosa::RescaleOp>(builder_->getUnknownLoc(), resultType, input, input_zp, output_zp, multiplier, shift, scale32, double_round, per_channel, input_unsigned, output_unsigned);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createReshapeOp(mlir::Type resultType, mlir::Value input, mlir::ArrayRef<int64_t> shape) {
//         auto op = builder_->create<mlir::tosa::ReshapeOp>(builder_->getUnknownLoc(), resultType, input, shape);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

// 	mlir::Operation* createResizeOp(mlir::Type resultType, mlir::Value input, mlir::ArrayRef<int64_t> newSize, mlir::ArrayRef<int64_t> offset, mlir::ArrayRef<int64_t> border, mlir::StringRef mode) {
//         auto op = builder_->create<mlir::tosa::ResizeOp>(builder_->getUnknownLoc(), resultType, input, newSize, offset, border, mode);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createReverseOp(mlir::Type resultType, mlir::Value input, uint32_t axes) {
//         auto op = builder_->create<mlir::tosa::ReverseOp>(builder_->getUnknownLoc(), resultType, input, axes);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createRsqrtOp(mlir::Type resultType, mlir::Value input) {
//         auto op = builder_->create<mlir::tosa::RsqrtOp>(builder_->getUnknownLoc(), resultType, input);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createScatterOp(mlir::Type resultType, mlir::Value input, mlir::Value indices, mlir::Value updates) {
//         auto op = builder_->create<mlir::tosa::ScatterOp>(builder_->getUnknownLoc(), resultType, updates, indices, input);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

// 	mlir::Operation* createSelectOp(mlir::Type resultType, mlir::Value cond, mlir::Value onTrue, mlir::Value onFalse) {
//         auto op = builder_->create<mlir::tosa::SelectOp>(builder_->getUnknownLoc(), resultType, cond, onTrue, onFalse);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createSigmoidOp(mlir::Type resultType, mlir::Value input) {
//         auto op = builder_->create<mlir::tosa::SigmoidOp>(builder_->getUnknownLoc(), resultType, input);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createSinOp(mlir::Type resultType, mlir::Value input) {
//         auto op = builder_->create<mlir::tosa::SinOp>(builder_->getUnknownLoc(), resultType, input);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createSliceOp(mlir::Type resultType, mlir::Value input, llvm::ArrayRef<int64_t> starts,
//                               llvm::ArrayRef<int64_t> sizes) {
//         auto op = builder_->create<mlir::tosa::SliceOp>(builder_->getUnknownLoc(), resultType, input, starts, sizes);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createSubOp(mlir::Type resultType, mlir::Value lhs, mlir::Value rhs) {
//         auto op = builder_->create<mlir::tosa::SubOp>(builder_->getUnknownLoc(), resultType, lhs, rhs);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createTableOp(mlir::Type resultType, mlir::Value input, mlir::Value table) {
//         auto op = builder_->create<mlir::tosa::TableOp>(builder_->getUnknownLoc(), resultType, input, table);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createTanhOp(mlir::Type resultType, mlir::Value input) {
//         auto op = builder_->create<mlir::tosa::TanhOp>(builder_->getUnknownLoc(), resultType, input);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createTileOp(mlir::Type resultType, mlir::Value input, mlir::Value multiples) {
//         auto op = builder_->create<mlir::tosa::TileOp>(builder_->getUnknownLoc(), resultType, input, multiples);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
// 	}

//     mlir::Operation* createTransposeConv2DOp(mlir::Type resultType, mlir::Value input, mlir::Value filter, mlir::Value bias,
//                                         llvm::ArrayRef<int64_t> outputShape, llvm::ArrayRef<int64_t> strides,
//                                          llvm::ArrayRef<int64_t> paddings,
//                                         mlir::Type accType) {
//         auto op = builder_->create<mlir::tosa::TransposeConv2DOp>(builder_->getUnknownLoc(), resultType, input, filter, bias,
//             paddings, strides, outputShape, accType, nullptr, false);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createTransposeOp(mlir::Type resultType, mlir::Value input, mlir::Value perm) {
//         auto op = builder_->create<mlir::tosa::TransposeOp>(builder_->getUnknownLoc(), resultType, input, perm);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }
    
//     mlir::Operation* createVariableOp(mlir::Type resultType, mlir::StringAttr name, mlir::Attribute initial_value) {
//         auto op = builder_->create<mlir::tosa::VariableOp>(builder_->getUnknownLoc(), name, resultType, initial_value);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createVariableReadOp(mlir::Type resultType, mlir::Value variable) {
//         auto op = builder_->create<mlir::tosa::VariableReadOp>(builder_->getUnknownLoc(), resultType, variable);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }
    
//     mlir::Operation* createVariableWriteOp(mlir::StringRef name, mlir::Value value) {
//         auto op = builder_->create<mlir::tosa::VariableWriteOp>(builder_->getUnknownLoc(), name, value);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createWhileOp(mlir::TypeRange resultType, mlir::ValueRange operands) {
//         auto op = builder_->create<mlir::tosa::WhileOp>(builder_->getUnknownLoc(), resultType, operands);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

//     mlir::Operation* createYieldOp(mlir::ValueRange results) {
//         auto op = builder_->create<mlir::tosa::YieldOp>(builder_->getUnknownLoc(), results);
//         opStack_->push(op.getOperation());
//         return op.getOperation();
//     }

};

} // namespace vkml