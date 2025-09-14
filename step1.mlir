#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @tosa_func_0(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = tensor.empty() : tensor<2x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%0 : tensor<2x3xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
  func.func @main() {
    %0 = tensor.empty() : tensor<2x3xf32>
    %1 = tensor.empty() : tensor<2x3xf32>
    %2 = call @tosa_func_0(%0, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return
  }
}

