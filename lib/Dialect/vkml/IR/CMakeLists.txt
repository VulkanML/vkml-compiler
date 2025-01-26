
add_mlir_dialect_library(MLIRvkmlDialect vkml.cpp 
    DEPENDS MLIRvkmlIncGen 
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms

    MLIRGPUDialect
    MLIRGPUTransforms

    MLIRSPIRVDialect
    MLIRSPIRVTarget
    MLIRSPIRVTransforms)