enable_language(CUDA)

#
# Architecture options:
#

# Create a new arch list that only contains arches that support CDP:
set(CUB_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
set(CUB_CUDA_ARCHITECTURES_RDC ${CUB_CUDA_ARCHITECTURES})
list(FILTER CUB_CUDA_ARCHITECTURES_RDC EXCLUDE REGEX "53|62|72|90")

message(STATUS "CUB_CUDA_ARCHITECTURES:     ${CUB_CUDA_ARCHITECTURES}")
message(STATUS "CUB_CUDA_ARCHITECTURES_RDC: ${CUB_CUDA_ARCHITECTURES_RDC}")

option(CUB_ENABLE_RDC_TESTS "Enable tests that require separable compilation." ON)
option(CUB_FORCE_RDC "Enable separable compilation on all targets that support it." OFF)

#
# Clang CUDA options 
#
if ("Clang" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unknown-cuda-version -Xclang=-fcuda-allow-variadic-functions")
endif ()
