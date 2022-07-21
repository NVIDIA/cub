# Enable or disable RDC for a CUDA target.
# Just using the CMake property won't work for our nvcxx builds, we need
# to manually specify flags.
# nvcc disables RDC by default, while nvc++ enables it. Thus this function
# must be called on all CUDA targets to get consistent RDC state across all
# platforms.
function(cub_set_rdc_state target_name enable)
  if ("NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
    if (enable)
      target_compile_options(${target_name} PRIVATE "-gpu=rdc")
    else()
      target_compile_options(${target_name} PRIVATE "-gpu=nordc")
    endif()
  else()
    set_target_properties(${target_name} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ${enable}
    )
  endif()
endfunction()
