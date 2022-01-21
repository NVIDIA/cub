# Enable RDC for a CUDA target. Encapsulates compiler hacks:
function(cub_enable_rdc_for_cuda_target target_name)
  if ("NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
    set_target_properties(${target_name} PROPERTIES
      COMPILE_FLAGS "-gpu=rdc"
    )
  else()
    set_target_properties(${target_name} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
    )
  endif()
endfunction()

# Add a set of descriptive labels to a test. CTest will use these to print a
# summary of time spent running each label's tests.
#
# This assumes that there's an executable target with the same name as the
# test, and that the executable target has been configured with either
# cub_set_target_properties or cub_clone_target_properties.
#
# Labels added are:
# - "cub"
# - C++ dialect
# - cub.dialect
function(cub_add_test_labels test_name)
  cub_get_target_property(config_dialect ${test_name} DIALECT)
  set(config_dialect "cpp${config_dialect}")

  set(test_labels
    cub
    ${config_dialect}
    cub.${config_dialect}
  )
  set_tests_properties(${test_name} PROPERTIES LABELS "${test_labels}")
endfunction()
