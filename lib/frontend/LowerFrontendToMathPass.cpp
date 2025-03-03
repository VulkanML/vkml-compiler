#include <mlir/Pass/Pass.h>
#include <mlir/IR/PatternMatch.h>
#include "mlir/Transforms/DialectConversion.h"

#include "frontend.h"
#include "frontend_passes.h"

using namespace mlir;


namespace frontend {
    struct LowerFrontendToMathPass : public PassWrapper<LowerFrontendToMathPass, OperationPass<mlir::ModuleOp>> {
		LowerFrontendToMathPass() = default;
        void runOnOperation() override;
        void getDependentDialects(DialectRegistry& registry) const final {
            registry.insert<mlir::func::FuncDialect>();
            registry.insert<mlir::arith::ArithDialect>();
            registry.insert<mlir::tensor::TensorDialect>();
            registry.insert<frontend::frontendDialect>();
        }
    };


std::unique_ptr<Pass> frontend::createLowerFrontendToMathPass() {
    return std::make_unique<LowerFrontendToMathPass>();
}

struct UnaryOpPattern: OpConversionPattern<frontend::unary_arith> {
    using OpConversionPattern<frontend::unary_arith>::OpConversionPattern;

    LogicalResult matchAndRewrite(frontend::unary_arith op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
       
 		auto isFloat = op.getDest().getType().getElementType().isa<mlir::FloatType>();   
        
        mlir::Operation* result;
        switch (op.getArithType()) {
		case frontend::UnaryArithEnum::ABS:
			result = rewriter.create<math::AbsFOp>(op.getLoc(), op.getOperand());
			break;
		case frontend::UnaryArithEnum::NEG:
			result = rewriter.create<arith::NegFOp>(op.getLoc(), op.getOperand());
			break;
		case frontend::UnaryArithEnum::SQRT:
			result = rewriter.create<math::SqrtOp>(op.getLoc(), op.getOperand());
			break;
        case frontend::UnaryArithEnum::SIN:
            result = rewriter.create<math::SinOp>(op.getLoc(), op.getOperand());
            break;
        case frontend::UnaryArithEnum::COS:
            result = rewriter.create<math::CosOp>(op.getLoc(), op.getOperand());
            break;
        case frontend::UnaryArithEnum::TAN:
            result = rewriter.create<math::TanOp>(op.getLoc(), op.getOperand());
            break;
        case frontend::UnaryArithEnum::ASIN:
            result = rewriter.create<math::AsinOp>(op.getLoc(), op.getOperand());
            break;
        case frontend::UnaryArithEnum::ACOS:
            result = rewriter.create<math::AcosOp>(op.getLoc(), op.getOperand());
            break;
        case frontend::UnaryArithEnum::ATAN:
            result = rewriter.create<math::AtanOp>(op.getLoc(), op.getOperand());
            break;
        case frontend::UnaryArithEnum::SINH:
            result = rewriter.create<math::SinhOp>(op.getLoc(), op.getOperand());
            break;
        case frontend::UnaryArithEnum::COSH:
            result = rewriter.create<math::CoshOp>(op.getLoc(), op.getOperand());
            break;
        case frontend::UnaryArithEnum::TANH:
            result = rewriter.create<math::TanhOp>(op.getLoc(), op.getOperand());
            break;
        case frontend::UnaryArithEnum::ASINH:
            result = rewriter.create<math::AsinhOp>(op.getLoc(), op.getOperand());
            break;
        case frontend::UnaryArithEnum::ACOSH:
            result = rewriter.create<math::AcoshOp>(op.getLoc(), op.getOperand());
            break;
        case frontend::UnaryArithEnum::ATANH:
            result = rewriter.create<math::AtanhOp>(op.getLoc(), op.getOperand());
            break;
		case frontend::UnaryArithEnum::LOG:
			result = rewriter.create<math::LogOp>(op.getLoc(), op.getOperand());
			break;
		case frontend::UnaryArithEnum::EXP:
			result = rewriter.create<math::ExpOp>(op.getLoc(), op.getOperand());
			break;
		case frontend::UnaryArithEnum::CEIL:
			result = rewriter.create<math::CeilOp>(op.getLoc(), op.getOperand());
			break;
		case frontend::UnaryArithEnum::FLOOR:
			result = rewriter.create<math::FloorOp>(op.getLoc(), op.getOperand());
			break;
		case frontend::UnaryArithEnum::ROUND:
			result = rewriter.create<math::RoundOp>(op.getLoc(), op.getOperand());
			break;

        case frontend::UnaryArithEnum::TRUNC:
            result = rewriter.create<math::TruncOp>(op.getLoc(), op.getOperand());
            break;
        default:
            return failure();
        }

        rewriter.replaceOp(op, result);
        return success();
    }
};

struct BinaryOpPattern: OpConversionPattern<frontend::binary_arith> {
    using OpConversionPattern<frontend::binary_arith>::OpConversionPattern;

    LogicalResult matchAndRewrite(frontend::binary_arith op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
   
        auto isFloat = op.getDest().getType().getElementType().isa<mlir::FloatType>();

        mlir::Operation* result;
        switch (op.getArithType()) {
        case frontend::BinaryArithEnum::ADD:
            result = rewriter.create<arith::AddFOp>(op.getLoc(), op.getLhs(), op.getRhs());
            break;

        case frontend::BinaryArithEnum::POW:
            result = rewriter.create<math::PowFOp>(op.getLoc(), op.getLhs(), op.getRhs());
            break;
        default:
            return failure();
        }

        rewriter.replaceOp(op, result);
        return success();
    }
};

struct ReductionOpPattern: OpConversionPattern<frontend::reduction> {
    using OpConversionPattern<frontend::reduction>::OpConversionPattern;

    LogicalResult matchAndRewrite(frontend::reduction op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
        frontend::ReductionEnum tmp = op.getReductionType();

        mlir::Operation* result;
        switch (op.getReductionType()) {
        //case frontend::ReductionEnum::REDUCE_SUM:
        //    result = rewriter.create<math::SumOp>(op.getLoc(), op.getOperand());
        //    break;
        //case frontend::ReductionEnum::REDUCE_MAX:
        //    result = rewriter.create<math::MaxOp>(op.getLoc(), op.getOperand());
        //    break;
        //case frontend::ReductionEnum::REDUCE_MIN:
        //    result = rewriter.create<math::MinOp>(op.getLoc(), op.getOperand());
        //    break;
        //case frontend::ReductionEnum::REDUCE_MEAN:
        //    result = rewriter.create<math::MeanOp>(op.getLoc(), op.getOperand());
        //    break;
        //case frontend::ReductionEnum::REDUCE_PROD:
        //    result = rewriter.create<math::ProdOp>(op.getLoc(), op.getOperand());
        //    break;
        //case frontend::ReductionEnum::REDUCE_L0:
        //    result = rewriter.create<math::L0Op>(op.getLoc(), op.getOperand());
        //    break;
        //case frontend::ReductionEnum::REDUCE_L1:
        //    result = rewriter.create<math::L1Op>(op.getLoc(), op.getOperand());
        //    break;
        default:
            return failure();
        }

        rewriter.replaceOp(op, result);
        return success();
    }
};

struct BitwiseOpPattern : OpConversionPattern<frontend::bitwise> {
    using OpConversionPattern<frontend::bitwise>::OpConversionPattern;

    LogicalResult matchAndRewrite(frontend::bitwise op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
        frontend::BitwiseEnum tmp = op.getBitwiseType();

        mlir::Operation* result;
        switch (op.getBitwiseType()) {
        case frontend::BitwiseEnum::BITWISE_AND:
            result = rewriter.create<arith::AndIOp>(op.getLoc(), op.getLhs(), op.getRhs());
            break;
        case frontend::BitwiseEnum::BITWISE_OR:
            result = rewriter.create<arith::OrIOp>(op.getLoc(), op.getLhs(), op.getRhs());
            break;
        case frontend::BitwiseEnum::BITWISE_XOR:
            result = rewriter.create<arith::XOrIOp>(op.getLoc(), op.getLhs(), op.getRhs());
            break;
        case frontend::BitwiseEnum::BITWISE_NOT:
          //  result = rewriter.create<arith::>(op.getLoc(), op.getOperand());
           // break;
        case frontend::BitwiseEnum::BITWISE_LEFT_SHIFT:
            result = rewriter.create<arith::ShLIOp>(op.getLoc(), op.getLhs(), op.getRhs());
            break;
        case frontend::BitwiseEnum::BITWISE_RIGHT_SHIFT:
            result = rewriter.create<arith::ShRSIOp>(op.getLoc(), op.getLhs(), op.getRhs());
            break;
        default:
            return failure();
        }

        rewriter.replaceOp(op, result);
        return success();
    }
};

struct LogicalOpPattern : OpConversionPattern<frontend::logical> {
    using OpConversionPattern<frontend::logical>::OpConversionPattern;

    LogicalResult matchAndRewrite(frontend::logical op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
        frontend::LogicalEnum tmp = op.getLogicalType();

        mlir::Operation* result;
        switch (op.getLogicalType()) {
        case frontend::LogicalEnum::LOGICAL_AND:
            result = rewriter.create<arith::AndIOp>(op.getLoc(), op.getLhs(), op.getRhs());
            break;
        case frontend::LogicalEnum::LOGICAL_OR:
            result = rewriter.create<arith::OrIOp>(op.getLoc(), op.getLhs(), op.getRhs());
            break;
        case frontend::LogicalEnum::LOGICAL_NOT: {
            // Create a constant of all 1's with same type as operand
            auto allOnes = rewriter.create<arith::ConstantOp>(
                op.getLoc(),
                op.getLhs().getType(),
                rewriter.getIntegerAttr(op.getLhs().getType(), -1)
            );
            // XOR with all 1's gives logical NOT
            result = rewriter.create<arith::XOrIOp>(op.getLoc(), op.getLhs(), allOnes);
            break;
        }
        default:
            return failure();
        }

        rewriter.replaceOp(op, result);
        return success();
    }
};

struct RelationalOpPattern : OpConversionPattern<frontend::relational> {
    using OpConversionPattern<frontend::relational>::OpConversionPattern;

    LogicalResult matchAndRewrite(frontend::relational op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
        frontend::RelationalEnum tmp = op.getRelationalType();

        mlir::Operation* result;
        switch (op.getRelationalType()) {
        case frontend::RelationalEnum::RELATIONAL_LT:
            result = rewriter.create<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::slt, op.getLhs(), op.getRhs());
            break;
        case frontend::RelationalEnum::RELATIONAL_LE:
            result = rewriter.create<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::sle, op.getLhs(), op.getRhs());
            break;
        case frontend::RelationalEnum::RELATIONAL_GT:
            result = rewriter.create<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::sgt, op.getLhs(), op.getRhs());
            break;
        case frontend::RelationalEnum::RELATIONAL_GE:
            result = rewriter.create<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::sge, op.getLhs(), op.getRhs());
            break;
        case frontend::RelationalEnum::RELATIONAL_EQ:
            result = rewriter.create<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::eq, op.getLhs(), op.getRhs());
            break;
        case frontend::RelationalEnum::RELATIONAL_NE:
            result = rewriter.create<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::ne, op.getLhs(), op.getRhs());
            break;
        default:
            return failure();
        }

        rewriter.replaceOp(op, result);
        return success();
    }
};

void LowerFrontendToMathPass::runOnOperation() {
    // Define the conversion target
    ConversionTarget target(getContext());
    //target.addIllegalDialect<frontend::frontendDialect>();
    target.addLegalDialect<mlir::BuiltinDialect>();
    target.addLegalDialect<mlir::math::MathDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::tensor::TensorDialect>();


    // Define the type converter
    TypeConverter typeConverter;

    // Define the rewrite patterns
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<UnaryOpPattern>(&getContext());
    patterns.add<BinaryOpPattern>(&getContext());
	patterns.add<ReductionOpPattern>(&getContext());
	patterns.add<BitwiseOpPattern>(&getContext());
	patterns.add<LogicalOpPattern>(&getContext());
	patterns.add<RelationalOpPattern>(&getContext());
    // Add your custom rewrite patterns here
    // patterns.insert<YourCustomPattern>(&getContext());

    // Apply the conversion
    if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}


} // end frontend namespace

