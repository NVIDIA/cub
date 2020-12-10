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
