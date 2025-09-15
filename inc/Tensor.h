#pragma once
#include <vector>
#include <string>
#include <type_traits>
#include <ostream>

#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

template<typename T>
class Tensor {
public:
    Tensor(const std::vector<int>& shape) : shape_(shape) {
        // Generate a unique symbolic ID for this tensor
        static int id_counter = 0;
        symbolic_id_ = "tensor_" + std::to_string(id_counter++);
        // Initialize mlir::Value here as needed
    }


    // Conversion constructor: Tensor<U> from Tensor<T>
    template<typename U,
        typename = std::enable_if_t<std::is_convertible_v<U, T>>>
    explicit Tensor(const Tensor<U>& other) {

        shape_ = other.getShape();
        symbolic_id_ = other.getSymbolicId(); // base ID reuse or regenerate below

        if constexpr (!std::is_same_v<T, U>) {
            // Optional: give a new symbolic id to distinguish cast
            symbolic_id_ += "_cast_" + std::string(typeid(U).name()) + "_to_" + std::string(typeid(T).name());
            // Placeholder for MLIR cast op creation (depends on your dialect / builder context):
            // value_ = createIntegerCast(builder, other.value_, /*from=*/typeOf<U>, /*to=*/typeOf<T>);
        }
        else {
            // Direct "copy" of underlying mlir::Value if semantics allow
            value_ = other.getValue();
        }
    }

    template<typename U, 
    typename = std::enable_if_t<std::is_arithmetic_v<U>>>
    explicit Tensor(const U& scalar): shape_{1}, symbolic_id_("scalar_" + std::to_string(scalar)) {
        // Initialize mlir::Value to represent the scalar
        // Placeholder for MLIR constant op creation (depends on your dialect / builder context):
        // value_ = createConstantOp(builder, scalar, typeOf<U>);
    }

    const std::vector<int>& getShape() const { return shape_; }
    std::string getSymbolicId() const { return symbolic_id_; }
	  mlir::Value getValue() const { return value_; }

private:
    std::vector<int> shape_;
    mlir::Value value_;
    std::string symbolic_id_;
    template<typename U>
    struct is_character_or_byte_or_bool : std::false_type {};

    template<>
    struct is_character_or_byte_or_bool<char> : std::true_type {};

    template<>
    struct is_character_or_byte_or_bool<unsigned char> : std::true_type {};

    template<>
    struct is_character_or_byte_or_bool<signed char> : std::true_type {};

    template<>
    struct is_character_or_byte_or_bool<std::byte> : std::true_type {};

    template<>
    struct is_character_or_byte_or_bool<bool> : std::true_type {};

    static std::vector<int> broadcastShapes(const std::vector<int>& shape1, const std::vector<int>& shape2) {
        std::vector<int> result;
        auto it1 = shape1.rbegin();
        auto it2 = shape2.rbegin();
        while (it1 != shape1.rend() || it2 != shape2.rend()) {
            int dim1 = (it1 != shape1.rend()) ? *it1 : 1;
            int dim2 = (it2 != shape2.rend()) ? *it2 : 1;
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                throw std::invalid_argument("Shapes cannot be broadcast together");
            }
            result.push_back(std::max(dim1, dim2));
            if (it1 != shape1.rend()) ++it1;
            if (it2 != shape2.rend()) ++it2;
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

public:

    // Arithmetic operators
    template<typename U, typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_signed_v<U>>>
    friend Tensor<U> operator+(const Tensor<U>& t) {
  	  return Tensor<U>(t.getShape());
    }

    template<typename U, typename = std::enable_if_t<std::is_arithmetic_v<U> && std::is_signed_v<U>>>
    friend Tensor<U> operator-(const Tensor<U>& t) {
		return Tensor<U>(t.getShape());
    }

    template<typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>>>
    friend Tensor<U> operator~(const Tensor<U>& t) {
		return Tensor<U>(t.getShape());
    }
    
    template<typename U, typename = std::enable_if_t<
        is_character_or_byte_or_bool<U>::value>>
    friend Tensor<bool> operator!(const Tensor<U>& t) {
        return Tensor<bool>(t.getShape());
    }

	template<typename U, typename V, typename = std::enable_if_t<
        std::is_arithmetic_v<U> && std::is_arithmetic_v<V> >>
    friend auto operator+(const Tensor<U>& lhs, const Tensor<V>& rhs) {
        using ResultType = std::common_type_t<U, V>;
        return Tensor<ResultType>(broadcastShapes(lhs.getShape(), rhs.getShape()));
	}

    template<typename U, typename V, typename = std::enable_if_t<
		std::is_arithmetic_v<U>&& std::is_arithmetic_v<V> >>
    friend auto operator-(const Tensor<U>& lhs, const Tensor<V>& rhs) {
        using ResultType = std::common_type_t<U, V>;
		return Tensor<ResultType>(broadcastShapes(lhs.getShape(), rhs.getShape()));
	}

	template<typename U, typename V, typename = std::enable_if_t<
		std::is_arithmetic_v<U>&& std::is_arithmetic_v<V> >>
    friend auto operator*(const Tensor<U>& lhs, const Tensor<V>& rhs) {
		using ResultType = std::common_type_t<U, V>;
		return Tensor<ResultType>(broadcastShapes(lhs.getShape(), rhs.getShape()));
	}

	template<typename U, typename V, typename = std::enable_if_t<
		std::is_arithmetic_v<U>&& std::is_arithmetic_v<V> >>
	friend auto operator/(const Tensor<U>& lhs, const Tensor<V>& rhs) {
		using ResultType = std::common_type_t<U, V>;
		return Tensor<ResultType>(broadcastShapes(lhs.getShape(), rhs.getShape()));
	}

	template<typename U, typename V, typename = std::enable_if_t<
		std::is_arithmetic_v<U>&& std::is_integral_v<V> >>
	friend auto operator%(const Tensor<U>& lhs, const Tensor<V>& rhs) {
		using ResultType = std::common_type_t<U, V>;
		return Tensor<ResultType>(broadcastShapes(lhs.getShape(), rhs.getShape()));
	}


  template<typename U, typename V, typename = std::enable_if_t<
		std::is_integral_v<U>&& std::is_integral_v<V>&& std::is_unsigned_v<U>&& std::is_unsigned_v<V>>>
	friend auto operator&(const Tensor<U>& lhs, const Tensor<V>& rhs) {
		using ResultType = std::common_type_t<U, V>;
		return Tensor<ResultType>(broadcastShapes(lhs.getShape(), rhs.getShape()));
	}

	template<typename U, typename V, typename = std::enable_if_t<
		std::is_integral_v<U>&& std::is_integral_v<V>&& std::is_unsigned_v<U>&& std::is_unsigned_v<V>>>
  friend auto operator|(const Tensor<U>& lhs, const Tensor<V>& rhs) {
        using ResultType = std::common_type_t<U, V>;
        return Tensor<ResultType>(broadcastShapes(lhs.getShape(), rhs.getShape()));
    }

	template<typename U, typename V, typename = std::enable_if_t<
        std::is_integral_v<U>&& std::is_integral_v<V>&& std::is_unsigned_v<U>&& std::is_unsigned_v<V>>>
  friend auto operator^(const Tensor<U>& lhs, const Tensor<V>& rhs) {
        using ResultType = std::common_type_t<U, V>;
        return Tensor<ResultType>(broadcastShapes(lhs.getShape(), rhs.getShape()));
	}


	template<typename U, typename V, typename = std::enable_if_t<
        std::is_integral_v<U>&& std::is_integral_v<V>&& std::is_unsigned_v<U>&& std::is_unsigned_v<V>>>
  friend auto operator<<(const Tensor<U>& lhs, const Tensor<V>& rhs) {
        using ResultType = std::common_type_t<U, V>;
        return Tensor<ResultType>(broadcastShapes(lhs.getShape(), rhs.getShape()));
	}

  template<typename U, typename V, typename = std::enable_if_t<
		std::is_integral_v<U>&& std::is_integral_v<V>&& std::is_unsigned_v<U>&& std::is_unsigned_v<V>>>
  friend auto operator>>(const Tensor<U>& lhs, const Tensor<V>& rhs) {
        using ResultType = std::common_type_t<U, V>;
		return Tensor<ResultType>(broadcastShapes(lhs.getShape(), rhs.getShape()));

	}


    // Increment and decrement operators    
  Tensor<T>& operator++() {        
      return *this;
  }

  Tensor<T>& operator--() {
      return *this;
  }

    // Comparison operators
	template<typename U, typename V, typename = std::enable_if_t <
		std::is_arithmetic_v<U>&& std::is_arithmetic_v<V> >>
  friend auto operator==(const Tensor<U>& lhs, const Tensor<V>& rhs) {
      return Tensor<bool>(broadcastShapes(lhs.getShape(), rhs.getShape()));
	}

	template<typename U, typename V, typename = std::enable_if_t <
		std::is_arithmetic_v<U>&& std::is_arithmetic_v<V> >>
  friend auto operator!=(const Tensor<T>& lhs, const Tensor<T>& rhs) {
      return Tensor<bool>(broadcastShapes(lhs.getShape(), rhs.getShape()));
	}

	template<typename U, typename V, typename = std::enable_if_t <
		std::is_arithmetic_v<U>&& std::is_arithmetic_v<V> >>
  friend auto operator<(const Tensor<U>& lhs, const Tensor<V>& rhs) {
        return Tensor<bool>(broadcastShapes(lhs.getShape(), rhs.getShape()));
  }

	template<typename U, typename V, typename = std::enable_if_t <
		std::is_arithmetic_v<U>&& std::is_arithmetic_v<V> >>
  friend auto operator>(const Tensor<U>& lhs, const Tensor<V>& rhs) {
		  return Tensor<bool>(broadcastShapes(lhs.getShape(), rhs.getShape()));
	}

	template<typename U, typename V, typename = std::enable_if_t <
		std::is_arithmetic_v<U>&& std::is_arithmetic_v<V> >>
  friend auto operator>=(const Tensor<U>& lhs, const Tensor<V>& rhs) {
		  return Tensor<bool>(broadcastShapes(lhs.getShape(), rhs.getShape()));
	}

    template<typename U, typename V, typename = std::enable_if_t <
		std::is_arithmetic_v<U>&& std::is_arithmetic_v<V> >>
    friend auto operator<=(const Tensor<U>& lhs, const Tensor<V>& rhs) {
        return Tensor<bool>(broadcastShapes(lhs.getShape(), rhs.getShape()));
	}

    template<typename U, typename V, typename = std::enable_if_t <
		std::is_arithmetic_v<U>&& std::is_arithmetic_v<V> >>
    friend auto operator&&(const Tensor<U>& lhs, const Tensor<V>& rhs) {
        return Tensor<bool>(broadcastShapes(lhs.getShape(), rhs.getShape()));
	}

    template<typename U, typename V, typename = std::enable_if_t <
        std::is_arithmetic_v<U> && std::is_arithmetic_v<V> >>
	friend auto operator||(const Tensor<U>& lhs, const Tensor<V>& rhs) {
        return Tensor<bool>(broadcastShapes(lhs.getShape(), rhs.getShape()));
	}

    // Subscript operator
    template<typename IndexType, typename = std::enable_if_t <
        std::is_integral_v<IndexType>&& std::is_unsigned_v<IndexType> >>
    Tensor<T>& operator[](const Tensor<IndexType>& index) {
        return *this;
    }

    template<typename IndexType, typename = std::enable_if_t <
        std::is_integral_v<IndexType>&& std::is_unsigned_v<IndexType> >>
    Tensor<T>& operator[](IndexType index) {
        return *this;
    }

    template <typename IndexType, typename = std::enable_if_t <
        std::is_integral_v<IndexType>&& std::is_signed_v<IndexType> >>
    const Tensor<T>& operator[](const Tensor<IndexType>& index) const {
        return *this;
    }

    template <typename IndexType, typename = std::enable_if_t <
        std::is_integral_v<IndexType>&& std::is_signed_v<IndexType> >>
    const Tensor<T>& operator[](IndexType index) const {
        return *this;
    }

    // Function call operator
    template<typename U, typename V>
    friend Tensor<std::common_type_t<U, V>> operator,(const Tensor<U>& lhs, const Tensor<V>& rhs) {
        using ResultType = std::common_type_t<U, V>;
        // Implement the desired behavior for the comma operator here.
        // For demonstration, return a Tensor with the shape of lhs.
        return Tensor<ResultType>(broadcastShapes(lhs.getShape(), rhs.getShape()));
    }

    // Assignment operator
    Tensor<T>& operator=(const Tensor& rhs) = delete;


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
        t.value_.print(llvm_os);
        llvm_os.flush();
        os << mlir_str;
        return os;
    }
    


    // Conversion operator
    // explicit operator bool() const;
    // explicit operator float() const;
    // explicit operator double() const;
    // explicit operator uint32_t() const;
    // explicit operator uint64_t() const;
    // explicit operator int32_t() const;
    // explicit operator int64_t() const;
    // explicit operator int8_t() const;
    // explicit operator uint8_t() const;

};

