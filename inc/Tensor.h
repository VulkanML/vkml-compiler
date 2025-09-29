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
#include <iostream>

#include "mlir/IR/Value.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"


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

    inline mlir::ArrayRef<int64_t> broadcastShapes(const mlir::ArrayRef<int64_t>& shape1, const mlir::ArrayRef<int64_t>& shape2) {
        mlir::SmallVector<int64_t> result;
        auto it1 = shape1.rbegin();
        auto it2 = shape2.rbegin();
        while (it1 != shape1.rend() || it2 != shape2.rend()) {
            int64_t dim1 = (it1 != shape1.rend()) ? *it1 : 1;
            int64_t dim2 = (it2 != shape2.rend()) ? *it2 : 1;
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                throw std::invalid_argument("Shapes cannot be broadcast together");
            }
            result.push_back(std::max(dim1, dim2));
            if (it1 != shape1.rend()) ++it1;
            if (it2 != shape2.rend()) ++it2;
        }
        std::reverse(result.begin(), result.end());
        return mlir::ArrayRef<long>(result.begin(), result.size()); 
    }
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

namespace vkml {
    class Compiler {
    private:
        mlir::MLIRContext context_;
        mlir::OpBuilder builder_;
        mlir::ModuleOp module_;
        static std::shared_ptr<Compiler> instance_;

        Compiler(): context_(), builder_(&context_) {
            context_.loadDialect<mlir::tosa::TosaDialect>();
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

    };

    std::shared_ptr<Compiler> Compiler::instance_ = nullptr;
    void dump(){
        printf("--- MLIR Module Dump ---\n");
        Compiler::getInstance()->getModule().dump();
    }
}



template<typename T>
class Tensor {
private:
    mlir::ArrayRef<int64_t> shape_;              // View into owned storage
    mlir::Operation* src_ = nullptr;
    bool external_ = false;
    
    mlir::Type elemType_;                       // Element type
    std::string symbolic_id_;
    mlir::RankedTensorType type_;
    std::shared_ptr<T> data_; 

    mutable mlir::tosa::VariableOp variableOp_;
    mutable mlir::tosa::VariableReadOp variableReadOp_;
    mutable mlir::tosa::VariableWriteOp variableWriteOp_;
    
public:

    Tensor(const mlir::ArrayRef<int64_t>& shape)
        : shape_(shape), src_(nullptr), 
          type_(mlir::RankedTensorType::get(shape_, 
            tensor_detail::cToMLIRType(vkml::Compiler::getInstance()->getContext(), 
            typeid(T)))), data_(nullptr)
    {
        variableReadOp_ = nullptr;
       variableWriteOp_ = nullptr;

        static int id_counter = 0; 
        symbolic_id_ = "tensor_" + std::to_string(id_counter++);

        {
            auto& builder = vkml::Compiler::getInstance()->getBuilder();
            auto loc = builder.getUnknownLoc();
            auto shapeAttr = builder.getI64TensorAttr(shape_);
            auto nameAttr = builder.getStringAttr(symbolic_id_);
            auto typeAttr = mlir::TypeAttr::get(type_);
            // Create an uninitialized variable (no initial value, no write).
            variableOp_ = builder.create<mlir::tosa::VariableOp>(loc, nameAttr, shapeAttr, typeAttr, mlir::Attribute{});
        }

        variableOp_.dump();
    }
    
    const std::vector<int64_t>& getShape() const { return shape_.vec(); }
    std::string getSymbolicId() const { return symbolic_id_; }

    // Conversion constructor: Tensor<U> from Tensor<T>
    template<typename U,
        typename = std::enable_if_t<std::is_convertible_v<U, T>>>
    explicit Tensor(const Tensor<U>& other) {
        
    }

    explicit Tensor(const T& scalar): shape_{1}, symbolic_id_("scalar_" + std::to_string(scalar)) {
         // Initialize mlir::Value to represent the scalar
    // Placeholder for MLIR constant op creation (depends on your dialect / builder context):
    }

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
    auto unaryOpHelper(){
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

public:

    // Binary arithmetic/logical operators (single template each)
    template<typename U, typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator+(const Tensor<U>& rhs) { return binaryOpHelper<mlir::tosa::AddOp>(*this, rhs); }
     
    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator-(const Tensor<U>& rhs) { return binaryOpHelper<mlir::tosa::SubOp>(*this, rhs);}
    
    template<typename U,  typename = std::enable_if_t<std::is_integral_v<U> && std::is_integral_v<T>>>
    auto operator/(const Tensor<U>& rhs) { return binaryOpHelper<mlir::tosa::IntDivOp>(*this, rhs); }
    
    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator*(const Tensor<U>& rhs) { 
        Tensor<uint8_t> scaleTensor({1});
        return ternaryOpHelper<mlir::tosa::MulOp>(*this, rhs, scaleTensor);
    }

    Tensor<T> operator+() const { return unaryOpHelper<mlir::tosa::AbsOp>(); }
   // Tensor<T> operator-() const { return unaryOpHelper<mlir::tosa::NegOp>(); }
    Tensor<T> operator~() const { return unaryOpHelper<mlir::tosa::BitwiseNotOp>(); }
    Tensor<T> operator!() const { return unaryOpHelper<mlir::tosa::LogicalNotOp>(); }
    
    
    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_integral_v<T>>>
    friend auto operator%(const Tensor<U>& lhs, const Tensor<T>& rhs) { 
        using CommonT = std::common_type_t<U, T>;
        auto& builder = vkml::Compiler::getInstance()->getBuilder();
        auto loc = builder.getUnknownLoc();
        auto ctx = vkml::Compiler::getInstance()->getContext();
        llvm::SmallVector<mlir::ShapedTypeComponents> inferred;
        if (mlir::failed(mlir::tosa::IntDivOp::inferReturnTypeComponents(ctx,
                std::optional<mlir::Location>{loc},
                mlir::ValueRange{lhs.read().getResult(), rhs.read().getResult()},
                /*attrs=*/mlir::DictionaryAttr{},
                /*properties=*/nullptr,
                /*regions=*/{},
                inferred))) {
            throw std::runtime_error("Op::inferReturnTypes failed");
        }
        auto elementType = tensor_detail::cToMLIRType(ctx, typeid(CommonT));
        auto resultType = mlir::RankedTensorType::get(inferred[0].getDims(), elementType);
        auto divOp = builder.create<mlir::tosa::IntDivOp>(loc
            , resultType, lhs.read().getResult(), rhs.read().getResult());
        auto mulOp = builder.create<mlir::tosa::MulOp>(loc
            , resultType, divOp.getResult(), rhs.read().getResult());
        auto subOp = builder.create<mlir::tosa::SubOp>(loc
            , resultType, lhs.read().getResult(), mulOp.getResult());
        auto output = Tensor<CommonT>(resultType.getShape());
        output.write(subOp.getResult());
        return std::move(output); 
    }
        
    template<typename U,  typename = std::enable_if_t<std::is_integral_v<U> && std::is_integral_v<T> && std::is_unsigned_v<U> && std::is_unsigned_v<T>>>
    auto operator&(const Tensor<U>& rhs) { return binaryOpHelper<mlir::tosa::BitwiseAndOp>(*this, rhs); }
    template<typename U,  typename = std::enable_if_t<std::is_integral_v<U> && std::is_integral_v<T> && std::is_unsigned_v<U> && std::is_unsigned_v<T>>>
    auto operator|(const Tensor<U>& rhs) { return binaryOpHelper<mlir::tosa::BitwiseOrOp>(*this, rhs); }
    template<typename U,  typename = std::enable_if_t<std::is_integral_v<U> && std::is_integral_v<T> && std::is_unsigned_v<U> && std::is_unsigned_v<T>>>
    auto operator^(const Tensor<U>& rhs) { return binaryOpHelper<mlir::tosa::BitwiseXorOp>(*this, rhs); }

    template<typename U,  typename = std::enable_if_t<std::is_integral_v<U> && std::is_integral_v<T> && std::is_unsigned_v<U> && std::is_unsigned_v<T>>>
    auto operator<<(const Tensor<U>& rhs) { return binaryOpHelper<mlir::tosa::LogicalLeftShiftOp>(*this, rhs); }
    template<typename U,  typename = std::enable_if_t<std::is_integral_v<U> && std::is_integral_v<T> && std::is_unsigned_v<U> && std::is_unsigned_v<T>>>
    auto operator>>(const Tensor<U>& rhs) { return binaryOpHelper<mlir::tosa::LogicalRightShiftOp>(*this, rhs); }

    Tensor<T>& operator++() {  
        
        return *this;
    }

    Tensor<T>& operator--() {
        return *this;
    }
    

    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator&&(const Tensor<U>& rhs) { return logicalOpHelper<mlir::tosa::LogicalAndOp>(*this, rhs); }
    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator||(const Tensor<U>& rhs) { return logicalOpHelper<mlir::tosa::LogicalOrOp>(*this, rhs); }

    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator==(const Tensor<U>& rhs) { return logicalOpHelper<mlir::tosa::EqualOp>(*this, rhs); }
    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator!=(const Tensor<U>& rhs) { return !(*this == rhs); }
    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator>(const Tensor<U>& rhs) { return logicalOpHelper<mlir::tosa::GreaterOp>(*this, rhs); }
    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator>=(const Tensor<U>& rhs) { return logicalOpHelper<mlir::tosa::GreaterEqualOp>(*this, rhs); }

    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
     auto operator<(const Tensor<U>& rhs) { return !(*this >= rhs); }
    template<typename U,  typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_arithmetic_v<T>>>
    auto operator<=(const Tensor<U>& rhs) { return !(*this > rhs); }


    // Subscript operator
    template<typename IndexType, typename = std::enable_if_t<std::is_integral_v<IndexType>&& std::is_unsigned_v<IndexType> >>
    Tensor<T>& operator[](const Tensor<IndexType>& index) {
        return *this;
    }

    template<typename IndexType, typename = std::enable_if_t<std::is_integral_v<IndexType>&& std::is_unsigned_v<IndexType> >>
    Tensor<T>& operator[](IndexType index) {
        return *this;
    }

    template <typename IndexType, typename = std::enable_if_t<std::is_integral_v<IndexType>&& std::is_signed_v<IndexType> >>
    const Tensor<T>& operator[](const Tensor<IndexType>& index) const {
        return *this;
    }

    template <typename IndexType, typename = std::enable_if_t<std::is_integral_v<IndexType>&& std::is_signed_v<IndexType> >>
    const Tensor<T>& operator[](IndexType index) const {
        return *this;
    }


    // Function call operator
    template<typename U>
    friend Tensor<std::common_type_t<U, T>> operator,(const Tensor<U>& lhs, const Tensor<T>& rhs) { return Tensor<std::common_type_t<U, T>>(tensor_detail::broadcastShapes(lhs.getShape(), rhs.getShape())); }


    // Assignment operator
    Tensor<T>& operator=(const Tensor& rhs) = delete;
    // Tensor<T>& operator=(const T& scalar) = delete;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        // Print a readable representation using the symbolic id and shape.
        os << t.symbolic_id_ << "(";
        for (std::size_t i = 0; i < t.shape_.size(); ++i) {
            if (i) os << "x";
            os << t.shape_[i];
        }
        os << ")";
        std::string mlir_str;
        llvm::raw_string_ostream llvm_os(mlir_str);
        
        t.type_.print(llvm_os);
        llvm_os.flush();
        os << mlir_str << " : ";

        
        // t.src_->print(llvm_os);
        llvm_os.flush();
        os << mlir_str;
        
        os << mlir_str;
        return os;
    }
    
};

