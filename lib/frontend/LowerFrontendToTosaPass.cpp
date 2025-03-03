#include <mlir/Pass/Pass.h>
#include <mlir/IR/PatternMatch.h>
#include "mlir/Transforms/DialectConversion.h"

#include "frontend.h"
#include "frontend_passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

using namespace mlir;

namespace frontend {

struct LowerFrontendToTosaPass : public PassWrapper<LowerFrontendToTosaPass, OperationPass<mlir::ModuleOp>> {
    void runOnOperation() override;
    void getDependentDialects(DialectRegistry& registry) const final {
        registry.insert<mlir::tosa::TosaDialect>();
        registry.insert<mlir::func::FuncDialect>();
        registry.insert<mlir::tensor::TensorDialect>();
        registry.insert<frontend::frontendDialect>();
    }
};

std::unique_ptr<Pass> createLowerFrontendToTosaPass() {
    return std::make_unique<LowerFrontendToTosaPass>();
}

struct UnaryOpToTosaPattern : public OpConversionPattern<frontend::unary_arith> {
    using OpConversionPattern<frontend::unary_arith>::OpConversionPattern;
    LogicalResult matchAndRewrite(
        frontend::unary_arith op, OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override {

        Operation* result;
        switch (op.getArithType()) {
        case frontend::UnaryArithEnum::ABS:
            result = rewriter.create<tosa::AbsOp>(op.getLoc(), op.getType(), op.getOperand());
            break;
        case frontend::UnaryArithEnum::NEG:
            result = rewriter.create<tosa::NegateOp>(op.getLoc(), op.getType(), op.getOperand());
            break;
        case frontend::UnaryArithEnum::SQRT: {
            // TOSA doesn't have sqrt directly, use rsqrt and reciprocal
            auto rsqrt = rewriter.create<tosa::RsqrtOp>(op.getLoc(), op.getType(), op.getOperand());
            // Create constant 1.0
            auto oneAttr = rewriter.getFloatAttr(
                op.getType().cast<TensorType>().getElementType(), 
                1.0);
            auto oneType = RankedTensorType::get({}, op.getType().cast<TensorType>().getElementType());
            auto oneConst = rewriter.create<tosa::ConstOp>(op.getLoc(), oneType, 
                DenseElementsAttr::get(oneType, oneAttr));
            // Reciprocal of rsqrt is sqrt
            result = rewriter.create<tosa::ReciprocalOp>(op.getLoc(), op.getType(), rsqrt.getResult());
            break;
        }
        case frontend::UnaryArithEnum::SIN:
            result = rewriter.create<tosa::SinOp>(op.getLoc(), op.getType(), op.getOperand());
            break;
        case frontend::UnaryArithEnum::COS:
            result = rewriter.create<tosa::CosOp>(op.getLoc(), op.getType(), op.getOperand());
            break;
       
        case frontend::UnaryArithEnum::TANH:
            result = rewriter.create<tosa::TanhOp>(op.getLoc(), op.getType(), op.getOperand());
            break;
       
        case frontend::UnaryArithEnum::EXP:
            result = rewriter.create<tosa::ExpOp>(op.getLoc(), op.getType(), op.getOperand());
            break;
        case frontend::UnaryArithEnum::LOG:
            result = rewriter.create<tosa::LogOp>(op.getLoc(), op.getType(), op.getOperand());
            break;
      
        case frontend::UnaryArithEnum::CEIL:
            result = rewriter.create<tosa::CeilOp>(op.getLoc(), op.getType(), op.getOperand());
            break;
        case frontend::UnaryArithEnum::FLOOR:
            result = rewriter.create<tosa::FloorOp>(op.getLoc(), op.getType(), op.getOperand());
            break;
        case frontend::UnaryArithEnum::ROUND: {
            // Implement round as floor(x + 0.5)
            auto halfType = RankedTensorType::get({}, op.getType().cast<TensorType>().getElementType());
            auto halfAttr = rewriter.getFloatAttr(
                op.getType().cast<TensorType>().getElementType(),
                0.5);
            auto halfConst = rewriter.create<tosa::ConstOp>(op.getLoc(), halfType,
                DenseElementsAttr::get(halfType, halfAttr));
            auto added = rewriter.create<tosa::AddOp>(
                op.getLoc(), op.getType(), op.getOperand(), halfConst);
            result = rewriter.create<tosa::FloorOp>(op.getLoc(), op.getType(), added);
            break;
        }
        }

        rewriter.replaceOp(op, result);
        return success();
    }
};

struct BinaryOpToTosaPattern : public OpConversionPattern<frontend::binary_arith> {
    using OpConversionPattern<frontend::binary_arith>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(
        frontend::binary_arith op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
        
        Operation* result = op;
        switch (op.getArithType()) {
        case frontend::BinaryArithEnum::ADD:
            result = rewriter.create<tosa::AddOp>(
                op.getLoc(), op.getType(), op.getLhs(), op.getRhs());
            break;
        case frontend::BinaryArithEnum::SUB:
            result = rewriter.create<tosa::SubOp>(
                op.getLoc(), op.getType(), op.getLhs(), op.getRhs());
            break;
        case frontend::BinaryArithEnum::MUL:
            result = rewriter.create<tosa::MulOp>(
                op.getLoc(), op.getType(), op.getLhs(), op.getRhs(), 
                rewriter.getI32IntegerAttr(0)); // No shift
            break;
        case frontend::BinaryArithEnum::DIV: {
            // TOSA doesn't have a DivOp - implement division as reciprocal followed by multiplication
            if (op.getType().cast<TensorType>().getElementType().isa<FloatType>()) {
                // For floating point, use reciprocal and multiply
                auto reciprocal = rewriter.create<tosa::ReciprocalOp>(
                    op.getLoc(), op.getRhs().getType(), op.getRhs());
                result = rewriter.create<tosa::MulOp>(
                    op.getLoc(), op.getType(), op.getLhs(), reciprocal.getResult(),
                    rewriter.getI32IntegerAttr(0)); // No shift
            }
            else {
                // For integer division, TOSA doesn't have a direct op
                // This would need a custom implementation
                return failure();
            }
            break;
        }
        case frontend::BinaryArithEnum::MAX:
            result = rewriter.create<tosa::MaximumOp>(
                op.getLoc(), op.getType(), op.getLhs(), op.getRhs());
            break;
        case frontend::BinaryArithEnum::MIN:
            result = rewriter.create<tosa::MinimumOp>(
                op.getLoc(), op.getType(), op.getLhs(), op.getRhs());
            break;
        case frontend::BinaryArithEnum::POW:
            result = rewriter.create<tosa::PowOp>(op.getLoc(), op.getType(), op.getLhs(), op.getRhs());
            break;
       
        }
        
        rewriter.replaceOp(op, result);
        return success();
    }
};

struct BitwiseOpToTosaPattern : public OpConversionPattern<frontend::bitwise> {
    using OpConversionPattern<frontend::bitwise>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(
        frontend::bitwise op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
        
        Operation* result = op;
        switch (op.getBitwiseType()) {
        case frontend::BitwiseEnum::BITWISE_AND:
            result = rewriter.create<tosa::BitwiseAndOp>(
                op.getLoc(), op.getType(), op.getLhs(), op.getRhs());
            break;
        case frontend::BitwiseEnum::BITWISE_OR:
            result = rewriter.create<tosa::BitwiseOrOp>(
                op.getLoc(), op.getType(), op.getLhs(), op.getRhs());
            break;
        case frontend::BitwiseEnum::BITWISE_XOR:
            result = rewriter.create<tosa::BitwiseXorOp>(
                op.getLoc(), op.getType(), op.getLhs(), op.getRhs());
            break;
        case frontend::BitwiseEnum::BITWISE_NOT:
            result = rewriter.create<tosa::BitwiseNotOp>(
                op.getLoc(), op.getType(), op.getLhs());
            break;
        case frontend::BitwiseEnum::BITWISE_LEFT_SHIFT:
            result = rewriter.create<tosa::LogicalLeftShiftOp>(
                op.getLoc(), op.getType(), op.getLhs(), op.getRhs());
            break;
        case frontend::BitwiseEnum::BITWISE_RIGHT_SHIFT:
            // For arithmetic right shift (preserves sign)
            result = rewriter.create<tosa::ArithmeticRightShiftOp>(
                op.getLoc(), op.getType(), op.getLhs(), op.getRhs(),
                rewriter.getBoolAttr(false)); // No round
            break;
      
        }
        
        rewriter.replaceOp(op, result);
        return success();
    }
};

struct LogicalOpToTosaPattern : public OpConversionPattern<frontend::logical> {
    using OpConversionPattern<frontend::logical>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        frontend::logical op, OpAdaptor adaptor, 
        ConversionPatternRewriter &rewriter) const override {
        
        Operation* result = op;
        switch (op.getLogicalType()) {
        case frontend::LogicalEnum::LOGICAL_AND:
            result = rewriter.create<tosa::LogicalAndOp>(
                op.getLoc(), op.getType(), op.getLhs(), op.getRhs());
            break;
        case frontend::LogicalEnum::LOGICAL_OR:
            result = rewriter.create<tosa::LogicalOrOp>(
                op.getLoc(), op.getType(), op.getLhs(), op.getRhs());
            break;        
        case frontend::LogicalEnum::LOGICAL_NOT:
            result = rewriter.create<tosa::LogicalNotOp>(
                op.getLoc(), op.getType(), op.getLhs());
            break;        
        }
        
        rewriter.replaceOp(op, result);
        return success();
    }
};

struct RelationalOpToTosaPattern : public OpConversionPattern<frontend::relational> {
    using OpConversionPattern<frontend::relational>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(
        frontend::relational op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
        
        Operation* result = op;
        switch (op.getRelationalType()) {
        case frontend::RelationalEnum::RELATIONAL_LT:
            // TOSA has GreaterOp, so swap operands for LT
            result = rewriter.create<tosa::GreaterOp>(
                op.getLoc(), op.getType(), op.getRhs(), op.getLhs());
            break;
        case frontend::RelationalEnum::RELATIONAL_LE:
            // TOSA has GreaterEqualOp, so swap operands for LE
            result = rewriter.create<tosa::GreaterEqualOp>(
                op.getLoc(), op.getType(), op.getRhs(), op.getLhs());
            break;
        case frontend::RelationalEnum::RELATIONAL_GT:
            result = rewriter.create<tosa::GreaterOp>(
                op.getLoc(), op.getType(), op.getLhs(), op.getRhs());
            break;
        case frontend::RelationalEnum::RELATIONAL_GE:
            result = rewriter.create<tosa::GreaterEqualOp>(
                op.getLoc(), op.getType(), op.getLhs(), op.getRhs());
            break;
        case frontend::RelationalEnum::RELATIONAL_EQ:
            result = rewriter.create<tosa::EqualOp>(
                op.getLoc(), op.getType(), op.getLhs(), op.getRhs());
            break;
        case frontend::RelationalEnum::RELATIONAL_NE:{
            // TOSA doesn't have NotEqual directly
            auto eq = rewriter.create<tosa::EqualOp>(
                op.getLoc(), op.getType(), op.getLhs(), op.getRhs());
            result = rewriter.create<tosa::LogicalNotOp>(
                op.getLoc(), op.getType(), eq);
            break; 
        }

        }
        
        rewriter.replaceOp(op, result);
        return success();
    }
};

struct ReductionOpToTosaPattern : public OpConversionPattern<frontend::reduction> {
    using OpConversionPattern<frontend::reduction>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(
        frontend::reduction op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
        
        // For TOSA reduction ops, we need to specify an axis directly as a uint32_t
        // Just use axis 0 (first dimension) as our reduction dimension
        uint32_t axis = 0; // Reduce along first dimension
		auto axisAttr = rewriter.getIntegerAttr(rewriter.getI32Type(), axis);
        Operation* result = op;
        switch (op.getReductionType()) {
        case frontend::ReductionEnum::REDUCE_SUM: {
            // Use the correct signature: Value input, uint32_t axis
            result = rewriter.create<tosa::ReduceSumOp>(
                op.getLoc(), op.getType(), op.getOperand(), axisAttr);
            break;
        }
        case frontend::ReductionEnum::REDUCE_MAX: {
            result = rewriter.create<tosa::ReduceMaxOp>(
                op.getLoc(), op.getType(), op.getOperand(), axisAttr);
            break;
        }
        case frontend::ReductionEnum::REDUCE_MIN: {
            result = rewriter.create<tosa::ReduceMinOp>(
                op.getLoc(), op.getType(), op.getOperand(), axisAttr);
            break;
        }
        case frontend::ReductionEnum::REDUCE_MEAN: {
            // TOSA doesn't have direct ReduceMean
            // Implement as sum followed by division by count
            auto sum = rewriter.create<tosa::ReduceSumOp>(
                op.getLoc(), op.getType(), op.getOperand(), axisAttr);
            
            // Get input type and calculate count of elements in the reduced dimension
            auto inputType = op.getOperand().getType().cast<RankedTensorType>();
            // Count is just the size of the dimension we're reducing
            int64_t count = inputType.getDimSize(axis);
            
            // Create a constant with the count
            auto countType = RankedTensorType::get({}, inputType.getElementType());
            Attribute countAttr;
            if (inputType.getElementType().isa<FloatType>()) {
                countAttr = rewriter.getFloatAttr(
                    inputType.getElementType(), 
                    static_cast<double>(count));
            } else {
                countAttr = rewriter.getIntegerAttr(
                    inputType.getElementType(), 
                    count);
            }
            auto countConst = rewriter.create<tosa::ConstOp>(
                op.getLoc(), countType,
                DenseElementsAttr::get(countType, countAttr));
            
            // Use reciprocal and multiply for division
            if (inputType.getElementType().isa<FloatType>()) {
                auto reciprocal = rewriter.create<tosa::ReciprocalOp>(
                    op.getLoc(), countConst.getType(), countConst);
                result = rewriter.create<tosa::MulOp>(
                    op.getLoc(), op.getType(), sum, reciprocal.getResult(),
                    rewriter.getI32IntegerAttr(0)); // No shift
            } else {
                // For integer types, approximating mean is more complex
                return failure();
            }
            break;
        }
        }
        
        rewriter.replaceOp(op, result);
        return success();
    }
};

void LowerFrontendToTosaPass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<mlir::tosa::TosaDialect>();
    target.addLegalDialect<mlir::tensor::TensorDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addIllegalDialect<frontend::frontendDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<UnaryOpToTosaPattern>(&getContext());
    patterns.add<BinaryOpToTosaPattern>(&getContext());
    patterns.add<BitwiseOpToTosaPattern>(&getContext());
    patterns.add<LogicalOpToTosaPattern>(&getContext());
    patterns.add<RelationalOpToTosaPattern>(&getContext());
    patterns.add<ReductionOpToTosaPattern>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}

} // namespace frontend
