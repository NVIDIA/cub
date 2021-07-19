if (NOT CUB_IN_THRUST) # Thrust has its own checks for this:
  # Test that we can use `find_package` on an installed CUB:
  add_test(
    NAME cub.test.cmake.test_install
    COMMAND "${CMAKE_COMMAND}"
      --log-level=VERBOSE
      -G "${CMAKE_GENERATOR}"
      -S "${CMAKE_CURRENT_SOURCE_DIR}/test_install"
      -B "${CMAKE_CURRENT_BINARY_DIR}/test_install"
      -D "CUB_BINARY_DIR=${CUB_BINARY_DIR}"
      -D "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      -D "CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
      -D "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  )
endif()

# Check that namespace macros are used correctly:
add_test(
  NAME cub.test.cmake.check_namespace
  COMMAND
    "${CMAKE_COMMAND}"
      -D "CUB_SOURCE_DIR=${CUB_SOURCE_DIR}"
      -P "${CMAKE_CURRENT_LIST_DIR}/check_namespace.cmake"
)
