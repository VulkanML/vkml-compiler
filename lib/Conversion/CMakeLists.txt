add_mlir_library(Conversion
  lowerFrontendToMLIRPass.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  FrontendDialect
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(Conversion MLIRFrontendIncGen)
add_dependencies(Conversion MLIRFrontendConversionPass)