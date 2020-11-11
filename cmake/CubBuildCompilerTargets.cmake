#
# This file defines the `cub_build_compiler_targets()` function, which
# creates the following interface targets:
#
# cub.compiler_interface
# - Interface target providing compiler-specific options needed to build
#   Thrust's tests, examples, etc.

function(cub_build_compiler_targets)
  set(cxx_compile_definitions)
  set(cxx_compile_options)

  if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    append_option_if_available("/W4" cxx_compile_options)

    append_option_if_available("/WX" cxx_compile_options)

    # Suppress overly-pedantic/unavoidable warnings brought in with /W4:
    # C4324: structure was padded due to alignment specifier
    append_option_if_available("/wd4324" cxx_compile_options)
    # C4127: conditional expression is constant
    # This can be fixed with `if constexpr` when available, but there's no way
    # to silence these pre-C++17.
    # TODO We should have per-dialect interface targets so we can leave these
    # warnings enabled on C++17:
    append_option_if_available("/wd4127" cxx_compile_options)
    # C4505: unreferenced local function has been removed
    # The CUDA `host_runtime.h` header emits this for
    # `__cudaUnregisterBinaryUtil`.
    append_option_if_available("/wd4505" cxx_compile_options)
    # C4706: assignment within conditional expression
    # MSVC doesn't provide an opt-out for this warning when the assignment is
    # intentional. Clang will warn for these, but suppresses the warning when
    # double-parentheses are used around the assignment. We'll let Clang catch
    # unintentional assignments and suppress all such warnings on MSVC.
    append_option_if_available("/wd4706" cxx_compile_options)

    # Some tests require /bigobj to fit everything into their object files:
    append_option_if_available("/bigobj" cxx_compile_options)
  else()
    append_option_if_available("-Werror" cxx_compile_options)
    append_option_if_available("-Wall" cxx_compile_options)
    append_option_if_available("-Wextra" cxx_compile_options)
    append_option_if_available("-Winit-self" cxx_compile_options)
    append_option_if_available("-Woverloaded-virtual" cxx_compile_options)
    append_option_if_available("-Wcast-qual" cxx_compile_options)
    append_option_if_available("-Wpointer-arith" cxx_compile_options)
    append_option_if_available("-Wunused-local-typedef" cxx_compile_options)
    append_option_if_available("-Wvla" cxx_compile_options)

    # Disable GNU extensions (flag is clang only)
    append_option_if_available("-Wgnu" cxx_compile_options)
    # Calling a variadic macro with zero args is a GNU extension until C++20,
    # but the THRUST_PP_ARITY macro is used with zero args. Need to see if this
    # is a real problem worth fixing.
    append_option_if_available("-Wno-gnu-zero-variadic-macro-arguments" cxx_compile_options)

    # This complains about functions in CUDA system headers when used with nvcc.
    append_option_if_available("-Wno-unused-function" cxx_compile_options)

    # CUB uses deprecated texture functions (cudaBindTexture, etc). These
    # need to be replaced, but silence the warnings for now.
    # This can be removed once NVIDIA/cub#191 is fixed.
    append_option_if_available("-Wno-deprecated-declarations" cxx_compile_options)
  endif()

  if ("GNU" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.3)
      # GCC 7.3 complains about name mangling changes due to `noexcept`
      # becoming part of the type system; we don't care.
      append_option_if_available("-Wno-noexcept-type" cxx_compile_options)
    endif()
  endif()

  if ("Intel" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    # Disable warning that inlining is inhibited by compiler thresholds.
    append_option_if_available("-diag-disable=11074" cxx_compile_options)
    append_option_if_available("-diag-disable=11076" cxx_compile_options)
  endif()

  if ("NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
    # Today:
    # * NVCC accepts CUDA C++ in .cu files but not .cpp files.
    # * NVC++ accepts CUDA C++ in .cpp files but not .cu files.
    # TODO: This won't be necessary in the future.
    list(APPEND cxx_compile_options -cppsuffix=cu)
  endif()

  add_library(cub.compiler_interface INTERFACE)

  foreach (cxx_option IN LISTS cxx_compile_options)
    target_compile_options(cub.compiler_interface INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:${cxx_option}>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVCXX>>:${cxx_option}>
      # Only use -Xcompiler with NVCC, not NVC++.
      #
      # CMake can't split genexs, so this can't be formatted better :(
      # This is:
      # if (using CUDA and CUDA_COMPILER is NVCC) add -Xcompiler=opt:
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcompiler=${cxx_option}>
    )
  endforeach()

  # Add these for both CUDA and CXX targets:
  target_compile_definitions(cub.compiler_interface INTERFACE
    ${cxx_compile_definitions}
  )

  # Promote warnings and display diagnostic numbers for nvcc:
  target_compile_options(cub.compiler_interface INTERFACE
    # If using CUDA w/ NVCC...
    # Display diagnostic numbers.
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcudafe=--display_error_number>
    # Promote warnings.
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcudafe=--promote_warnings>
    # Don't complain about deprecated GPU targets.
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Wno-deprecated-gpu-targets>
  )
endfunction()
