<<<<<<< HEAD

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${HEADERS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
file(GLOB_RECURSE GEN_HEADERS ${CMAKE_CURRENT_BINARY_DIR}) 
=======
add_subdirectory(Dialect)
add_subdirectory(Conversion)
>>>>>>> 7acffda (fifth commit)
