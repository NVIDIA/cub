enable_language(CUDA)

#
# Architecture options:
#
set(all_archs 35 37 50 52 53 60 61 62 70 72 75 80 86)
set(arch_message "CUB: Explicitly enabled compute architectures:")
set(ptx_targets)

# Thrust sets up the architecture flags in CMAKE_CUDA_FLAGS already. Just
# reuse them if possible. After we transition to CMake 3.18 CUDA_ARCHITECTURE
# target properties this will need to be updated.
if (CUB_IN_THRUST)
  # Configure to use all flags from thrust:
  set(CMAKE_CUDA_FLAGS "${THRUST_CUDA_FLAGS_BASE} ${THRUST_CUDA_FLAGS_NO_RDC}")

  # Update the enabled architectures list from thrust
  foreach (arch IN LISTS all_archs)
    if (THRUST_ENABLE_COMPUTE_${arch})
      set(CUB_ENABLE_COMPUTE_${arch} True)
      string(APPEND arch_message " sm_${arch}")
      list(APPEND ptx_targets "${arch}0")
    else()
      set(CUB_ENABLE_COMPUTE_${arch} False)
    endif()
  endforeach()

  # Otherwise create cache options and build the flags ourselves:
else() # NOT CUB_IN_THRUST

  # Find the highest arch:
  list(SORT all_archs)
  list(LENGTH all_archs max_idx)
  math(EXPR max_idx "${max_idx} - 1")
  list(GET all_archs ${max_idx} highest_arch)

  set(option_init OFF)
  if ("NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
    set(option_init ON)
  endif()
  option(CUB_DISABLE_ARCH_BY_DEFAULT
    "If ON, then all compute architectures are disabled on the initial CMake run."
    ${option_init}
  )

  set(option_init ON)
  if (CUB_DISABLE_ARCH_BY_DEFAULT)
    set(option_init OFF)
  endif()

  set(arch_flags)
  set(num_archs_enabled 0)
  foreach (arch IN LISTS all_archs)
    option(CUB_ENABLE_COMPUTE_${arch}
      "Enable code generation for sm_${arch}."
      ${option_init}
    )

    if (CUB_ENABLE_COMPUTE_${arch})
      math(EXPR num_archs_enabled "${num_archs_enabled} + 1")

      if ("NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
        if (NOT ${num_archs_enabled} EQUAL 1)
          message(FATAL_ERROR
            "NVC++ does not support compilation for multiple device architectures "
            "at once."
          )
        endif()
        set(arch_flag "-gpu=cc${arch}")
      else()
        string(APPEND arch_flags " -gencode arch=compute_${arch},code=sm_${arch}")
      endif()

      string(APPEND arch_message " sm_${arch}")
      list(APPEND ptx_targets "${arch}0")
    endif()
  endforeach()

  if (NOT "NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
    option(CUB_ENABLE_COMPUTE_FUTURE
      "Enable code generation for tests for compute_${highest_arch}"
      ${option_init}
    )
    if (CUB_ENABLE_COMPUTE_FUTURE)
      string(APPEND arch_flags
        " -gencode arch=compute_${highest_arch},code=compute_${highest_arch}"
      )
      string(APPEND arch_message " compute_${highest_arch}")
    endif()
  endif()

  # TODO Once CMake 3.18 is required, use the CUDA_ARCHITECTURE target props
  string(APPEND CMAKE_CUDA_FLAGS "${arch_flags}")
endif()

message(STATUS ${arch_message})

# Define CUB_PTX_TARGETS to restrict the number of kernel tunings that are
# instantiated. This is distinct from the logic in cub-config.cmake, since
# CUB detects PTX arches after finding the CUB package.
# After 11.5, __CUDA_ARCH_LIST__ should be available, and CUB will automatically
# use that instead.
if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND
    CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.5)
  list(JOIN ptx_targets "\\," ptx_targets)
  target_compile_definitions(_CUB_CUB INTERFACE "CUB_PTX_TARGETS=${ptx_targets}")
endif()

#
# RDC options:
#

# RDC is off by default in NVCC and on by default in NVC++. Turning off RDC
# isn't currently supported by NVC++. So, we default to RDC off for NVCC and
# RDC on for NVC++.
set(option_init OFF)
if ("NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  set(option_init ON)
endif()

option(CUB_ENABLE_TESTS_WITH_RDC
  "Build all CUB tests with RDC; tests that require RDC are not affected by this option."
  ${option_init}
)

option(CUB_ENABLE_EXAMPLES_WITH_RDC
  "Build all CUB examples with RDC; examples which require RDC are not affected by this option."
  ${option_init}
)

# Check for RDC/SM compatibility and error/warn if necessary
set(no_rdc_archs 53 62 72)
set(rdc_supported True)
foreach (arch IN LISTS no_rdc_archs)
  if (CUB_ENABLE_COMPUTE_${arch})
    set(rdc_supported False)
    break()
  endif()
endforeach()

set(rdc_opts
  CUB_ENABLE_TESTS_WITH_RDC
  CUB_ENABLE_EXAMPLES_WITH_RDC
)
set(rdc_requested False)
foreach (rdc_opt IN LISTS rdc_opts)
  if (${rdc_opt})
    set(rdc_requested True)
    break()
  endif()
endforeach()

if (rdc_requested AND NOT rdc_supported)
  string(JOIN ", " no_rdc ${no_rdc_archs})
  string(JOIN "\n" opts ${rdc_opts})
  message(FATAL_ERROR
    "Architectures {${no_rdc}} do not support RDC and are incompatible with "
    "these options:\n${opts}"
  )
endif()
