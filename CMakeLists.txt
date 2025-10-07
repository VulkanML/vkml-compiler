cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0048 NEW)
project(vkml-compiler VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


option(TOOLS "ENABLE TESTING" ON)

set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_INCLUDE_RUNTIMES OFF CACHE BOOL "")
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBPFM OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBXML2 OFF CACHE BOOL "")
set(LLVM_ENABLE_RTTI ON CACHE BOOL "")
set(LLVM_ENABLE_EH ON CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
set(LLVM_ENABLE_CRASH_OVERRIDES OFF CACHE BOOL "")
set(LLVM_ENABLE_BACKTRACES OFF CACHE BOOL "")
set(LLVM_ENABLE_OCAMLDOC OFF CACHE BOOL "")
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "")
set(LLVM_ENABLE_WARNINGS OFF CACHE BOOL "")
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "")
set(LLVM_ENABLE_IDE ON CACHE BOOL "")
set(LLVM_BUILD_RUNTIMES OFF CACHE BOOL "")
set(LLVM_BUILD_TOOLS OFF CACHE BOOL "")
set(LLVM_BUILD_UTILS ON CACHE BOOL "")
set(MLIR_ENABLE_EXECUTION_ENGINE OFF CACHE BOOL "")
set(MLIR_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "")
set(LLVM_BUILD_INSTRUMENTED_COVERAGE OFF CACHE BOOL "")
set(LLVM_TARGETS_TO_BUILD "" CACHE STRING "")
set(LLVM_EXPERIMENTAL_TARGETS_TO_BUILD "SPIRV" CACHE STRING "")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llvm/llvm EXCLUDE_FROM_ALL)
set(MLIR_CMAKE_DIR ${CMAKE_BINARY_DIR}/lib/cmake/mlir)
set(LLVM_CMAKE_DIR ${CMAKE_BINARY_DIR}/llvm/llvm/lib/cmake/llvm)
list(APPEND CMAKE_PREFIX_PATH ${MLIR_CMAKE_DIR})
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(TableGen)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/tablegen)

add_subdirectory(lib)

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${HEADERS})
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/lib/*.cpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

file(GLOB_RECURSE GEN_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/*.h.inc")
source_group("generated" FILES ${GEN_HEADERS})
file(GLOB_RECURSE GEN_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/*.cpp.inc")
source_group("generated" FILES ${GEN_SOURCES})

add_library(vkml-compiler STATIC 
     ${HEADERS} ${GEN_HEADERS} ${SOURCES} ${GEN_SOURCES} )

target_link_libraries(vkml-compiler 
 	# "MLIRTensorAllExtensions" 
 	# "MLIRTensorDialect" 
 	# "MLIRTensorInferTypeOpInterfaceImpl"
 	# "MLIRTensorMeshShardingExtensions" 
 	# "MLIRTensorTilingInterfaceImpl"
	"MLIRTensorToLinalg" 
# 	"MLIRTensorTransformOps"
# 	"MLIRTensorToSPIRV" 
# 	"MLIRTensorTransforms"
	"MLIRFuncDialect"
	"MLIRTosaDialect" 
	"MLIRTosaTransforms"
	"MLIRMLProgramDialect"
	"MLIRTosaToMLProgram"
	"MLIRTosaToLinalg" 
# 	"MLIRTosaToArith"
	"MLIRTosaToTensor"
	"MLIRTosaToSCF"
# 	"MLIRTosaShardingInterfaceImpl"
	"MLIRGPUDialect"
	"MLIRGPUTransforms"
# 	"MLIRSPIRVDialect"
# 	"MLIRAffineToStandard"
# 	"MLIRSCFToGPU"
# 	"MLIRBufferizationToMemRef"
)

target_include_directories(vkml-compiler 
	PRIVATE
		${CMAKE_CURRENT_BINARY_DIR}/tablegen
)

target_include_directories(vkml-compiler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)

set_property(GLOBAL PROPERTY VKML_COMPILER_HEADERS ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})


add_executable(vkml-test-app main.cpp)
target_link_libraries(vkml-test-app PRIVATE vkml-compiler)
target_include_directories(vkml-test-app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)

if(TOOLS)
	add_subdirectory(tools)
endif()

