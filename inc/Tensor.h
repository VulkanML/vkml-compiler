#pragma once
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>
#include <vector>
#include <string>
#include <type_traits>
#include <cstddef>
#include <ostream>

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm-c/Core.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/raw_os_ostream.h"


// Implementation details for Tensor utilities
namespace tensor_detail {
    template<typename U>
    struct is_character_or_byte_or_bool : std::bool_constant<
        std::is_same_v<U, char> ||
        std::is_same_v<U, unsigned char> ||
        std::is_same_v<U, signed char> ||
        std::is_same_v<U, std::byte> ||
        std::is_same_v<U, bool>
    > {};

    
    static auto cToMLIRType = [](mlir::MLIRContext* ctx, const std::type_info& type) -> mlir::Type {
        if (type == typeid(float)) {
            return mlir::Float32Type::get(ctx);
        } else if (type == typeid(double)) {
            return mlir::Float64Type::get(ctx);
        } else if (type == typeid(char)) {
            return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signed);
        } else if (type == typeid(unsigned char)) {
            return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);
        } else if (type == typeid(int32_t)) {
            return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
        } else if (type == typeid(int64_t)) {
            return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
        } else if (type == typeid(uint32_t)) {
            return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Unsigned);
        } else if (type == typeid(uint64_t)) {
            return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
        } else if (type == typeid(bool)) {
            return mlir::IntegerType::get(ctx, 1, mlir::IntegerType::Unsigned);
        } else {
            throw std::invalid_argument("Unsupported type for MLIR conversion");
        }
    };

}

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace vkml {
    class Compiler {
    private:
        mlir::MLIRContext context_;
        mlir::OpBuilder builder_;
        mlir::ModuleOp module_;
        mlir::func::FuncOp func_;
        static std::shared_ptr<Compiler> instance_;

        Compiler(): context_(), builder_(&context_) {
            context_.loadDialect<mlir::tosa::TosaDialect>();
            context_.loadDialect<mlir::func::FuncDialect>();
            module_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
            builder_.setInsertionPointToStart(module_.getBody());
        }
    public:
        
        Compiler(const Compiler&) = delete;
        Compiler& operator=(const Compiler&) = delete;

        static std::shared_ptr<Compiler> getInstance() {
            if(instance_.get() == nullptr)
                instance_ = std::shared_ptr<Compiler>(new Compiler());
            return instance_;
        }

        mlir::MLIRContext* getContext() { return &context_; }
        mlir::OpBuilder& getBuilder() { return builder_; }
        mlir::ModuleOp getModule() { return module_; }
        mlir::Location getUnknownLoc() { return builder_.getUnknownLoc(); }

        mlir::func::FuncOp getOrCreateMain() {
            if (auto func_ = module_.lookupSymbol<mlir::func::FuncOp>("main"))
                return func_;
            auto loc = builder_.getUnknownLoc();
            auto fnType = builder_.getFunctionType({}, {});
            func_ = builder_.create<mlir::func::FuncOp>(loc, "main", fnType);
            auto *entry = func_.addEntryBlock();
             mlir::OpBuilder::InsertionGuard g(builder_);
            builder_.setInsertionPointToStart(entry);
            builder_.create<mlir::func::ReturnOp>(loc);
            return func_;
        }
        void setInsertionIntoMain() {
            auto func = getOrCreateMain();
            auto &block = func.getBody().front();
            mlir::Operation *terminator = block.getTerminator();
            builder_.setInsertionPoint(terminator); 
        }

        mlir::tosa::VariableOp createVariable(mlir::RankedTensorType type,
                                            llvm::ArrayRef<int64_t> shape,
                                            llvm::StringRef name) {
            auto loc = builder_.getUnknownLoc();
            auto shapeAttr = builder_.getI64TensorAttr(shape);
            auto nameAttr = builder_.getStringAttr(name);
            auto typeAttr = mlir::TypeAttr::get(type);
            return builder_.create<mlir::tosa::VariableOp>(loc, nameAttr, shapeAttr, typeAttr, mlir::Attribute{});
        }

    };

    std::shared_ptr<Compiler> Compiler::instance_ = nullptr;

    inline void dump(){
        if(mlir::failed( Compiler::getInstance()->getModule().verify() )){
            throw std::runtime_error("Module verification failed");
        }
      
        mlir::OpPrintingFlags flags;
        flags.printGenericOpForm();
        flags.shouldPrintDebugInfo();
        Compiler::getInstance()->getModule().print(llvm::outs(), flags);    
    }
   
}



template<typename T>
class Tensor {
private:
    std::vector<int64_t> shapeStorage_;          // Owns the shape memory
    mlir::ArrayRef<int64_t> shape_;              // View into owned storage
    mlir::Operation* src_ = nullptr;
    bool external_ = false;
    
    std::string symbolic_id_;
    mlir::RankedTensorType type_;
    std::shared_ptr<T> data_; 

    mutable mlir::tosa::VariableOp variableOp_;
    mutable mlir::tosa::VariableReadOp variableReadOp_;
    mutable mlir::tosa::VariableWriteOp variableWriteOp_;
   
    
public:

    // Allow different template instantiations to access private members
    template<typename> friend class Tensor;

    // Convenience constructor to disambiguate brace-init usage
    Tensor(std::initializer_list<int64_t> dims) : Tensor(mlir::ArrayRef<int64_t>(dims.begin(), dims.size())) {}

        Tensor(const mlir::ArrayRef<int64_t>& shape)
                : shapeStorage_(shape.begin(), shape.end()), shape_(shapeStorage_), src_(nullptr), 
          type_(mlir::RankedTensorType::get(shape_, 
            tensor_detail::cToMLIRType(vkml::Compiler::getInstance()->getContext(), 
        typeid(T)))), data_(nullptr), variableWriteOp_(nullptr), variableReadOp_(nullptr)
    {
        variableReadOp_ = nullptr;
        variableWriteOp_ = nullptr;

        static int id_counter = 0; 
        symbolic_id_ = "tensor_" + std::to_string(id_counter++);
        auto compiler = vkml::Compiler::getInstance();
        compiler->setInsertionIntoMain();
        variableOp_ = compiler->createVariable(type_, shape_, symbolic_id_);
        
    }
    
    const std::vector<int64_t>& getShape() const { return shapeStorage_; }
    std::string getSymbolicId() const { return symbolic_id_; }

    // Conversion constructor: Tensor<U> from Tensor<T>
    template<typename U,
        typename = std::enable_if_t<std::is_convertible_v<U, T>>>
    explicit Tensor(const Tensor<U>& other) {
        auto readOp = other.read();
        auto& builder = vkml::Compiler::getInstance()->getBuilder();
        auto loc = builder.getUnknownLoc();
        auto ctx = vkml::Compiler::getInstance()->getContext();
        llvm::SmallVector<mlir::ShapedTypeComponents> inferred;
        if (mlir::failed(
            mlir::tosa::CastOp::inferReturnTypeComponents(ctx,
                std::optional<mlir::Location>{loc},
                mlir::ValueRange{readOp.getResult()},
                /*attrs=*/mlir::DictionaryAttr{},
                /*properties=*/nullptr,
                /*regions=*/{},
                inferred))) {
            throw std::runtime_error("Op::inferReturnTypes failed");
        }   
        auto elementType = tensor_detail::cToMLIRType(ctx, typeid(T));
        auto resultType = mlir::RankedTensorType::get(inferred[0].getDims(), elementType);
        auto castOp = builder.create<mlir::tosa::CastOp>(loc, resultType, readOp.getResult());
        this->write(castOp.getResult());
    this->shapeStorage_ = other.getShape();
    this->shape_ = shapeStorage_;
        this->type_ = resultType;
        this->symbolic_id_ = other.getSymbolicId() + "_casted";
        this->variableReadOp_ = nullptr;
    }

    explicit Tensor(const T& scalar, const mlir::ArrayRef<int64_t>& shape): Tensor<T>(shape){
        // Initialize mlir::Value to represent the scalar
         
    }

private:
    inline mlir::tosa::VariableReadOp read() const {
        if(variableReadOp_ == nullptr){
            auto& builder = vkml::Compiler::getInstance()->getBuilder();
            auto loc = builder.getUnknownLoc();
            variableReadOp_ = builder.create<mlir::tosa::VariableReadOp>(loc, type_, variableOp_.getNameAttr());
        }
        return variableReadOp_;
    }
    
    inline void write(mlir::Value newValue)  {
        if(variableWriteOp_ == nullptr) {
            auto& builder = vkml::Compiler::getInstance()->getBuilder();
            auto loc = builder.getUnknownLoc();
            variableWriteOp_ = builder.create<mlir::tosa::VariableWriteOp>(loc, variableOp_.getNameAttr(), newValue);
        } 
    } 

    template <typename Op, typename U, typename V, typename W>
    static auto ternaryOpHelper(const Tensor<U>& a, const Tensor<V>& b, const Tensor<W>& c) {
        using commonT = std::common_type_t<U, V>;
        auto& builder = vkml::Compiler::getInstance()->getBuilder();
        auto loc = builder.getUnknownLoc();
        auto ctx = vkml::Compiler::getInstance()->getContext();
        llvm::SmallVector<mlir::ShapedTypeComponents> inferred;
        if (mlir::failed(
            Op::inferReturnTypeComponents(ctx,
                std::optional<mlir::Location>{loc},
                mlir::ValueRange{a.read().getResult(), b.read().getResult(), c.read().getResult()},
                /*attrs=*/mlir::DictionaryAttr{},
                /*properties=*/nullptr,
                /*regions=*/{},
                inferred))) {
            throw std::runtime_error("Op::inferReturnTypes failed");
        }
        auto elementType = tensor_detail::cToMLIRType(ctx, typeid(commonT));
        auto resultType = mlir::RankedTensorType::get(inferred[0].getDims(), elementType);
        auto op = builder.create<Op>(loc, resultType, a.read().getResult(), b.read().getResult(), c.read().getResult());
        auto output = Tensor<commonT>(resultType.getShape());
        output.write(op.getResult());
        return std::move(output);
    }

    template<typename Op, typename U , typename V>
    static auto binaryOpHelper(const Tensor<U>& lhs, const Tensor<V>& rhs){
        using commonT = std::common_type_t<U, V>;
        auto &builder = vkml::Compiler::getInstance()->getBuilder();
        auto loc = builder.getUnknownLoc();
        auto ctx = vkml::Compiler::getInstance()->getContext();
        llvm::SmallVector<mlir::ShapedTypeComponents> inferred;
        if (mlir::failed(
            Op::inferReturnTypeComponents(ctx,
                std::optional<mlir::Location>{loc},
                mlir::ValueRange{lhs.read().getResult(), rhs.read().getResult()},
                /*attrs=*/mlir::DictionaryAttr{},
                /*properties=*/nullptr,
                /*regions=*/{},
                inferred))) {
            throw std::runtime_error("Op::inferReturnTypes failed");
        }

        auto elementType = tensor_detail::cToMLIRType(ctx, typeid(commonT));
        auto resultType = mlir::RankedTensorType::get(inferred[0].getDims(), elementType);
        auto op = builder.create<Op>(loc, resultType, lhs.read().getResult(), rhs.read().getResult());
        auto output = Tensor<commonT>(resultType.getShape());
        output.write(op.getResult());
        return std::move(output);
    }

    template<typename Op, typename U, typename V>
    static auto logicalOpHelper(const Tensor<U>& lhs, const Tensor<V>& rhs){
        auto &builder = vkml::Compiler::getInstance()->getBuilder();
        auto loc = builder.getUnknownLoc();
        auto ctx = vkml::Compiler::getInstance()->getContext();
        llvm::SmallVector<mlir::ShapedTypeComponents> inferred;
        if (mlir::failed(
            Op::inferReturnTypeComponents(ctx,
                std::optional<mlir::Location>{loc},
                mlir::ValueRange{lhs.read().getResult(), rhs.read().getResult()},
                /*attrs=*/mlir::DictionaryAttr{},
                /*properties=*/nullptr,
                /*regions=*/{},
                inferred))) {
            throw std::runtime_error("Op::inferReturnTypes failed");
        }

        auto elementType = tensor_detail::cToMLIRType(ctx, typeid(bool));
        auto resultType = mlir::RankedTensorType::get(inferred[0].getDims(), elementType);
        auto op = builder.create<Op>(loc, resultType, lhs.read().getResult(), rhs.read().getResult());
        auto output = Tensor<bool>(resultType.getShape());
        output.write(op.getResult());
        return std::move(output);
    }

    template<typename Op>
    auto unaryOpHelper() const {
        auto &builder = vkml::Compiler::getInstance()->getBuilder();
        auto loc = builder.getUnknownLoc();
        auto ctx = vkml::Compiler::getInstance()->getContext();
        llvm::SmallVector<mlir::ShapedTypeComponents> inferred;
        if (mlir::failed(
            Op::inferReturnTypeComponents(ctx,
                std::optional<mlir::Location>{loc},
                mlir::ValueRange{this->read().getResult()},
                /*attrs=*/mlir::DictionaryAttr{},
                /*properties=*/nullptr,
                /*regions=*/{},
                inferred))) {
            throw std::runtime_error("Op::inferReturnTypes failed");
        }

        auto elementType = tensor_detail::cToMLIRType(ctx, typeid(T));
        auto resultType = mlir::RankedTensorType::get(inferred[0].getDims(), elementType);
        auto op = builder.create<Op>(loc, resultType, this->read().getResult());
        auto output = Tensor<T>(resultType.getShape());
        output.write(op.getResult());
        return std::move(output);
    }

    template<bool isIncrement>
    void applyInPlaceIncrementDecrement(){
        static_assert(std::is_arithmetic_v<T>, "Increment/decrement only supported for arithmetic tensor types");
        auto &builder = vkml::Compiler::getInstance()->getBuilder();
        auto loc = builder.getUnknownLoc();
        auto ctx = vkml::Compiler::getInstance()->getContext();

        mlir::Type elemTy = tensor_detail::cToMLIRType(ctx, typeid(T));
        // Use shape {1} to ease creating DenseElementsAttr uniformly.
        mlir::RankedTensorType oneTy = mlir::RankedTensorType::get({1}, elemTy);
        mlir::Attribute elementAttr;
        if constexpr (std::is_floating_point_v<T>) {
            if (elemTy.isF32()) elementAttr = builder.getF32FloatAttr(1.0f);
            else if (elemTy.isF64()) elementAttr = builder.getF64FloatAttr(1.0);
            else elementAttr = builder.getF32FloatAttr(1.0f);
        } else {
            elementAttr = builder.getIntegerAttr(elemTy, 1);
        }
        std::array<mlir::Attribute,1> attrArr{elementAttr};
        auto valueAttr = mlir::DenseElementsAttr::get(oneTy, llvm::ArrayRef<mlir::Attribute>(attrArr));
        auto constOp = builder.create<mlir::tosa::ConstOp>(loc, oneTy, valueAttr);
        Tensor<T> oneTensor({1});
        oneTensor.write(constOp.getResult());

        // Use existing arithmetic helpers (broadcast will occur via TOSA rules)
        if constexpr (isIncrement) {
            auto updated = (*this) + oneTensor; // returns new tensor
            this->write(updated.read().getResult());
        } else {
            auto updated = (*this) - oneTensor;
            this->write(updated.read().getResult());
        }
    }
    

public:

    // Binary arithmetic/logical operators (single template each)
    template<typename U, typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator+(const Tensor<U>& rhs) const { return binaryOpHelper<mlir::tosa::AddOp>(*this, rhs); }
     
    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator-(const Tensor<U>& rhs) const { return binaryOpHelper<mlir::tosa::SubOp>(*this, rhs);}
    
    // Unified division operator: integer -> IntDivOp, floating -> reciprocal * mul
    template<typename U, typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator/(const Tensor<U>& rhs) const {
        if constexpr (std::is_integral_v<U> && std::is_integral_v<T>) {
            return binaryOpHelper<mlir::tosa::IntDivOp>(*this, rhs);
        } else if constexpr (std::is_floating_point_v<U> && std::is_floating_point_v<T>) {
            auto recip = rhs.template unaryOpHelper<mlir::tosa::ReciprocalOp>();
            return (*this) * recip; // reuse MulOp path
        } else {
            static_assert(std::is_same_v<U, void>, "Mixed integral/floating division not supported");
        }
    }
    
    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator*(const Tensor<U>& rhs) const { 
        Tensor<uint8_t> scaleTensor({1}); // scale/shift tensor for tosa.mul signature
        return ternaryOpHelper<mlir::tosa::MulOp>(*this, rhs, scaleTensor);
    }

    Tensor<T> operator+() const { return unaryOpHelper<mlir::tosa::AbsOp>(); }
   // Tensor<T> operator-() const { return unaryOpHelper<mlir::tosa::NegOp>(); }
    Tensor<T> operator~() const { return unaryOpHelper<mlir::tosa::BitwiseNotOp>(); }
    Tensor<T> operator!() const { return unaryOpHelper<mlir::tosa::LogicalNotOp>(); }
    
    
    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<T> && std::is_integral_v<U>>>
    auto operator%(const Tensor<U>& rhs) const { 
        // Modulo: a % b = a - (a / b) * b  (integer arithmetic semantics)
        // Reuse existing operator overloads to build IR safely.
        return *this - ((*this / rhs) * rhs);
    }
        
    template<typename U,  typename = std::enable_if_t<std::is_integral_v<U> && std::is_integral_v<T> && std::is_unsigned_v<U> && std::is_unsigned_v<T>>>
    auto operator&(const Tensor<U>& rhs) const { return binaryOpHelper<mlir::tosa::BitwiseAndOp>(*this, rhs); }
    template<typename U,  typename = std::enable_if_t<std::is_integral_v<U> && std::is_integral_v<T> && std::is_unsigned_v<U> && std::is_unsigned_v<T>>>
    auto operator|(const Tensor<U>& rhs) const { return binaryOpHelper<mlir::tosa::BitwiseOrOp>(*this, rhs); }
    template<typename U,  typename = std::enable_if_t<std::is_integral_v<U> && std::is_integral_v<T> && std::is_unsigned_v<U> && std::is_unsigned_v<T>>>
    auto operator^(const Tensor<U>& rhs) const { return binaryOpHelper<mlir::tosa::BitwiseXorOp>(*this, rhs); }

    template<typename U,  typename = std::enable_if_t<std::is_integral_v<U> && std::is_integral_v<T> && std::is_unsigned_v<U> && std::is_unsigned_v<T>>>
    auto operator<<(const Tensor<U>& rhs) const { return binaryOpHelper<mlir::tosa::LogicalLeftShiftOp>(*this, rhs); }
    template<typename U,  typename = std::enable_if_t<std::is_integral_v<U> && std::is_integral_v<T> && std::is_unsigned_v<U> && std::is_unsigned_v<T>>>
    auto operator>>(const Tensor<U>& rhs) const { return binaryOpHelper<mlir::tosa::LogicalRightShiftOp>(*this, rhs); }

   

    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator&&(const Tensor<U>& rhs) const { return logicalOpHelper<mlir::tosa::LogicalAndOp>(*this, rhs); }
    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator||(const Tensor<U>& rhs) const { return logicalOpHelper<mlir::tosa::LogicalOrOp>(*this, rhs); }

    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator==(const Tensor<U>& rhs) const { return logicalOpHelper<mlir::tosa::EqualOp>(*this, rhs); }
    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator!=(const Tensor<U>& rhs) const { return !(*this == rhs); }
    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator>(const Tensor<U>& rhs) const { return logicalOpHelper<mlir::tosa::GreaterOp>(*this, rhs); }
    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator>=(const Tensor<U>& rhs) const { return logicalOpHelper<mlir::tosa::GreaterEqualOp>(*this, rhs); }

    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
     auto operator<(const Tensor<U>& rhs) const { return !(*this >= rhs); }
    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator<=(const Tensor<U>& rhs) const { return !(*this > rhs); }
    
    friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& t){
        // Print a readable representation using the symbolic id and shape.
        os << t.symbolic_id_ << "(";
        for (std::size_t i = 0; i < t.shape_.size(); ++i) {
            if (i) os << "x";
            os << t.shape_[i];
        }
        os << ")";

        // Print the MLIR type.
        std::string mlir_str;
        llvm::raw_string_ostream llvm_os(mlir_str);
        t.type_.print(llvm_os);
        llvm_os.flush();
        os << " " << mlir_str;
        return os;
    }
  
    Tensor<T>& operator++() { applyInPlaceIncrementDecrement</*isIncrement=*/true>(); return *this; }

    Tensor<T>& operator--() { applyInPlaceIncrementDecrement</*isIncrement=*/false>(); return *this; }

    // Postfix increment
    Tensor<T> operator++(int) { applyInPlaceIncrementDecrement</*isIncrement=*/true>(); return *this;}

    // Postfix decrement
    Tensor<T> operator--(int) { applyInPlaceIncrementDecrement</*isIncrement=*/false>(); return *this; }

    // Subscript operator
    // Option A implementation: return a new Tensor<T> representing a slice along the first dimension.
    // Currently only supports a single integral index into the first dimension; rank is reduced by 1.
    // If original shape is [D0, D1, ..., Dn] result shape is [D1, ..., Dn].
    template<typename IndexType, typename = std::enable_if_t<std::is_integral_v<IndexType> && std::is_unsigned_v<IndexType>>>
    Tensor<T> operator[](IndexType index) const {
        if(shape_.empty()) {
            throw std::out_of_range("Cannot index into a rank-0 tensor");
        }
        if(index >= static_cast<IndexType>(shape_[0])) {
            throw std::out_of_range("Index out of bounds for first dimension");
        }
        auto &builder = vkml::Compiler::getInstance()->getBuilder();
        auto loc = builder.getUnknownLoc();
        auto ctx = vkml::Compiler::getInstance()->getContext();

        // Build start and size vectors (rank = original rank)
        const int rank = static_cast<int>(shape_.size());
        std::vector<int64_t> start(rank, 0);
        start[0] = static_cast<int64_t>(index);
        std::vector<int64_t> size(shape_.begin(), shape_.end());
        size[0] = 1; // single slice along first dim

        // Create TOSA shape operands as tosa.const (since we don't yet wrap tosa.shape values).
        // Represent them as !tosa.shape<rank> which expects ranked shape type.
        auto shapeType = mlir::tosa::shapeType::get(ctx, rank);

        // Helper lambda to materialize a shape const using DenseIntElementsAttr.
        auto makeShapeConst = [&](const std::vector<int64_t>& vals){
            // Represent as a 1D tensor of i64 for the DenseIntElementsAttr then rely on implicit conversion in builder.
            auto int64Ty = builder.getIntegerType(64, mlir::IntegerType::Signed);
            auto tensorTy = mlir::RankedTensorType::get({static_cast<long>(vals.size())}, int64Ty);
            llvm::SmallVector<int64_t> copy(vals.begin(), vals.end());
            auto denseAttr = mlir::DenseIntElementsAttr::get(tensorTy, copy);
            return builder.create<mlir::tosa::ConstShapeOp>(loc, shapeType, denseAttr);
        };

        auto startOp = makeShapeConst(start);
        auto sizeOp  = makeShapeConst(size);

        // Infer result type: it should have same rank as input (TOSA slice keeps rank) then we manually drop dim0 for API semantics.
        // We'll still emit the slice with original rank result; after emission we wrap a Tensor<T> whose shape excludes the first dim.
        llvm::SmallVector<mlir::ShapedTypeComponents> inferred;
        if(mlir::failed(mlir::tosa::SliceOp::inferReturnTypeComponents(ctx,
             std::optional<mlir::Location>{loc},
             mlir::ValueRange{this->read().getResult(), startOp.getResult(), sizeOp.getResult()},
             mlir::DictionaryAttr{}, nullptr, {}, inferred))) {
            throw std::runtime_error("SliceOp::inferReturnTypeComponents failed");
        }
        auto elementType = tensor_detail::cToMLIRType(ctx, typeid(T));
        auto fullResultType = mlir::RankedTensorType::get(inferred[0].getDims(), elementType);
        auto sliceOp = builder.create<mlir::tosa::SliceOp>(loc, fullResultType, this->read().getResult(), startOp.getResult(), sizeOp.getResult());

        // Build reduced-rank shape for API (drop first dim)
        std::vector<int64_t> reducedShape;
        reducedShape.reserve(shape_.size() - 1);
        for(size_t i=1;i<shape_.size();++i) reducedShape.push_back(shape_[i]);
        Tensor<T> result(reducedShape);
        result.write(sliceOp.getResult());
        return result;
    }

    // Overload taking a Tensor index (must be scalar tensor with compatible unsigned integral element). Not yet implemented: will throw.
    template<typename IndexTensorType, typename = std::enable_if_t<std::is_integral_v<IndexTensorType> && std::is_unsigned_v<IndexTensorType>>>
    Tensor<T> operator[](const Tensor<IndexTensorType>& /*indexTensor*/) const {
        throw std::runtime_error("Tensor index operand not yet supported for operator[]");
    }


    // // Function call operator
    // template<typename U>
    // friend Tensor<std::common_type_t<U, T>> operator,(const Tensor<U>& lhs, const Tensor<T>& rhs) { return Tensor<std::common_type_t<U, T>>(tensor_detail::broadcastShapes(lhs.getShape(), rhs.getShape())); }


    // Assignment operator
    Tensor<T>& operator=(const Tensor& rhs) = delete;

private:

};

