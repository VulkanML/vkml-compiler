// #include "compiler.h"

// int main() {
//     vkml::Compiler compiler;
//     vkml::TensorOps tensorOps(compiler);
//     vkml::TosaOps tosaOps(compiler);

//     // Create operations in logical order - they'll be stacked automatically
//     auto t1 = tensorOps.createEmptyOp(
//         mlir::Float32Type::get(compiler.getContext()),
//         {2, 3}
//     );

//     auto t2 = tensorOps.createEmptyOp(
//         mlir::Float32Type::get(compiler.getContext()),
//         {2, 3}
//     );

//     // Create TOSA add operation
//     auto add = tosaOps.createAddOp(
//         t1->getResult(0).getType(),
//         t1->getResult(0),
//         t2->getResult(0)
//     );
    
//     // Run the transformation passes
//     compiler.runPasses();

//     return 0;
// }

#include "Tensor.h"
#include <iostream>

int main() {
    Tensor<float> tensor_0({2, 3});
    Tensor<float> tensor_1({1, 3});
    auto result = tensor_0 + tensor_1;
    auto result2 = result -tensor_0;

    
    std::cout << "Tensor 0: " << tensor_0 << std::endl;
    std::cout << "Tensor 1: " << tensor_1 << std::endl;
    // std::cout << "Result Tensor: " << result << std::endl;
    // std::cout << "Result2 Tensor: " << result2 << std::endl;
    
    vkml::dump();
    return 0;
 }