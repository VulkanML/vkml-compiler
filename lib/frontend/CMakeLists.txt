get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)


add_mlir_dialect_library(MLIRfrontendDialect 
    frontend.cpp
    LowerFrontendToBackendPass.cpp
    LowerFrontendToMathPass.cpp
    LowerFrontendToTosaPass.cpp
    LowerFrontendToMLIRPass.cpp
    
    DEPENDS frontend_header MLIRfrontendIncGen 

    DISABLE_INSTALL

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms

    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRTensorDialect
    MLIRMathDialect
    ${conversion_libs} ${extension_libs} ${translation_libs}
)