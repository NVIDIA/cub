enable_language(CUDA)

if (NOT CUB_IN_THRUST)
  message(FATAL_ERROR
    "Building CUB as a standalone project is no longer supported. "
    "Use the Thrust repo instead.")
endif()

set(CUB_CUDA_FLAGS_BASE "${THRUST_CUDA_FLAGS_BASE}")
set(CUB_CUDA_FLAGS_RDC "${THRUST_CUDA_FLAGS_RDC}")
set(CUB_CUDA_FLAGS_NO_RDC "${THRUST_CUDA_FLAGS_NO_RDC}")

# Update the enabled architectures list from thrust
foreach (arch IN LISTS THRUST_KNOWN_COMPUTE_ARCHS)
  if (THRUST_ENABLE_COMPUTE_${arch})
    set(CUB_ENABLE_COMPUTE_${arch} True)
    string(APPEND arch_message " sm_${arch}")
  else()
    set(CUB_ENABLE_COMPUTE_${arch} False)
  endif()
endforeach()

message(STATUS ${arch_message})

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


# 
# Clang CUDA options 
#
if ("Clang" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  set(CUB_CUDA_FLAGS_BASE "${CUB_CUDA_FLAGS_BASE} -Wno-unknown-cuda-version -Xclang=-fcuda-allow-variadic-functions")
endif()


# By default RDC is not used:
set(CMAKE_CUDA_FLAGS "${CUB_CUDA_FLAGS_BASE} ${CUB_CUDA_FLAGS_NO_RDC}")
