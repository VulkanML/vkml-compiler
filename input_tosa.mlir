module {
  func.func @main() {
    %0 = tensor.empty() : tensor<2x3xf32>
    %1 = tensor.empty() : tensor<2x3xf32>
    %0 = tosa.add %0, %1 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return
  }
}