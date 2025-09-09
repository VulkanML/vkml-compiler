#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
module attributes {gpu.container_module} {
  func.func @tosa_func_0(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_memref %arg1 : tensor<2x3xf32> to memref<2x3xf32, strided<[?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg0 : tensor<2x3xf32> to memref<2x3xf32, strided<[?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
    %c1_0 = arith.constant 1 : index
    %2 = affine.apply #map(%c2)[%c0, %c1]
    %3 = affine.apply #map(%c3)[%c0, %c1]
    gpu.launch_func  @tosa_func_0_kernel::@tosa_func_0_kernel blocks in (%2, %3, %c1_0) threads in (%c1_0, %c1_0, %c1_0)  args(%c1 : index, %c0 : index, %1 : memref<2x3xf32, strided<[?, ?], offset: ?>>, %0 : memref<2x3xf32, strided<[?, ?], offset: ?>>, %alloc : memref<2x3xf32>)
    %4 = bufferization.to_tensor %alloc : memref<2x3xf32> to tensor<2x3xf32>
    return %4 : tensor<2x3xf32>
  }
  gpu.module @tosa_func_0_kernel {
    gpu.func @tosa_func_0_kernel(%arg0: index, %arg1: index, %arg2: memref<2x3xf32, strided<[?, ?], offset: ?>>, %arg3: memref<2x3xf32, strided<[?, ?], offset: ?>>, %arg4: memref<2x3xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = affine.apply #map1(%block_id_x)[%arg0, %arg1]
      %1 = affine.apply #map1(%block_id_y)[%arg0, %arg1]
      %2 = memref.load %arg2[%0, %1] : memref<2x3xf32, strided<[?, ?], offset: ?>>
      %3 = memref.load %arg3[%0, %1] : memref<2x3xf32, strided<[?, ?], offset: ?>>
      %4 = arith.addf %2, %3 : f32
      memref.store %4, %arg4[%0, %1] : memref<2x3xf32>
      gpu.return
    }
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

