module {
  func.func @tosa_func_0(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_memref %arg1 : tensor<2x3xf32> to memref<2x3xf32, strided<[?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg0 : tensor<2x3xf32> to memref<2x3xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    scf.for %arg2 = %c0 to %c2 step %c1 {
      scf.for %arg3 = %c0 to %c3 step %c1 {
        %3 = memref.load %1[%arg2, %arg3] : memref<2x3xf32, strided<[?, ?], offset: ?>>
        %4 = memref.load %0[%arg2, %arg3] : memref<2x3xf32, strided<[?, ?], offset: ?>>
        %5 = arith.addf %3, %4 : f32
        memref.store %5, %alloc[%arg2, %arg3] : memref<2x3xf32>
      }
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

