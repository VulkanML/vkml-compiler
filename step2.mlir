#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @tosa_func_0(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = bufferization.to_memref %arg1 : tensor<2x3xf32> to memref<2x3xf32, strided<[?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg0 : tensor<2x3xf32> to memref<2x3xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %0 : memref<2x3xf32, strided<[?, ?], offset: ?>>, memref<2x3xf32, strided<[?, ?], offset: ?>>) outs(%alloc : memref<2x3xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.addf %in, %in_0 : f32
      linalg.yield %3 : f32
    }
    %2 = bufferization.to_tensor %alloc : memref<2x3xf32> to tensor<2x3xf32>
    return %2 : tensor<2x3xf32>
  }
  func.func @main() {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    %0 = bufferization.to_tensor %alloc : memref<2x3xf32> to tensor<2x3xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    %1 = bufferization.to_tensor %alloc_0 : memref<2x3xf32> to tensor<2x3xf32>
    %2 = call @tosa_func_0(%0, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return
  }
}

