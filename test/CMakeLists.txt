# Some tests always build with RDC, so make sure that the sm_XX flags are
# compatible. See note in CubCudaConfig.cmake.
# TODO once we're using CUDA_ARCHITECTURES, we can setup non-rdc fallback
# tests to build for non-rdc arches. But for now, all files in a given directory
# must build with the same `CMAKE_CUDA_FLAGS` due to CMake constraints around
# how CUDA_FLAGS works.
set(CMAKE_CUDA_FLAGS "${CUB_CUDA_FLAGS_BASE} ${CUB_CUDA_FLAGS_RDC}")

# The function below reads the filepath `src`, extracts the %PARAM% comments,
# and fills `labels_var` with a list of `label1_value1.label2_value2...`
# strings, and puts the corresponding `DEFINITION=value1:DEFINITION=value2`
# entries into `defs_var`.
#
# See the README.md file in this directory for background info.
function(cub_get_test_params src labels_var defs_var)
  file(READ "${src}" file_data)
  set(param_regex "//[ ]+%PARAM%[ ]+([^ ]+)[ ]+([^ ]+)[ ]+([^\n]*)")

  string(REGEX MATCHALL
    "${param_regex}"
    matches
    "${file_data}"
  )

  set(variant_labels)
  set(variant_defs)

  foreach(match IN LISTS matches)
    string(REGEX MATCH
      "${param_regex}"
      unused
      "${match}"
    )

    set(def ${CMAKE_MATCH_1})
    set(label ${CMAKE_MATCH_2})
    set(values "${CMAKE_MATCH_3}")
    string(REPLACE ":" ";" values "${values}")

    # Build lists of test name suffixes (labels) and preprocessor definitions
    # (defs) containing the cartesian product of all param values:
    if (NOT variant_labels)
      foreach(value IN LISTS values)
        list(APPEND variant_labels ${label}_${value})
      endforeach()
    else()
      set(tmp_labels)
      foreach(old_label IN LISTS variant_labels)
        foreach(value IN LISTS values)
          list(APPEND tmp_labels ${old_label}.${label}_${value})
        endforeach()
      endforeach()
      set(variant_labels "${tmp_labels}")
    endif()

    if (NOT variant_defs)
      foreach(value IN LISTS values)
        list(APPEND variant_defs ${def}=${value})
      endforeach()
    else()
      set(tmp_defs)
      foreach(old_def IN LISTS variant_defs)
        foreach(value IN LISTS values)
          list(APPEND tmp_defs ${old_def}:${def}=${value})
        endforeach()
      endforeach()
      set(variant_defs "${tmp_defs}")
    endif()
  endforeach()

  set(${labels_var} "${variant_labels}" PARENT_SCOPE)
  set(${defs_var} "${variant_defs}" PARENT_SCOPE)
endfunction()

# Create meta targets that build all tests for a single configuration:
foreach(cub_target IN LISTS CUB_TARGETS)
  cub_get_target_property(config_prefix ${cub_target} PREFIX)
  set(config_meta_target ${config_prefix}.tests)
  add_custom_target(${config_meta_target})
  add_dependencies(${config_prefix}.all ${config_meta_target})
endforeach()

file(GLOB test_srcs
  RELATIVE "${CUB_SOURCE_DIR}/test"
  CONFIGURE_DEPENDS
  test_*.cu
)

file(GLOB catch2_test_srcs
  RELATIVE "${CUB_SOURCE_DIR}/test"
  CONFIGURE_DEPENDS
  catch2_test_*.cu
)

## cub_add_test
#
# Add a test executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the test
#   target. Useful for post-processing target information.
# test_name: The name of the test minus "<config_prefix>.test." For example,
#   testing/vector.cu will be "vector", and testing/cuda/copy.cu will be
#   "cuda.copy".
# test_src: The source file that implements the test.
# cub_target: The reference cub target with configuration information.
#
function(cub_add_test target_name_var test_name test_src cub_target)
  cub_get_target_property(config_prefix ${cub_target} PREFIX)

  # The actual name of the test's target:
  set(test_target ${config_prefix}.test.${test_name})
  set(${target_name_var} ${test_target} PARENT_SCOPE)

  # Related target names:
  set(config_meta_target ${config_prefix}.tests)
  set(test_meta_target cub.all.test.${test_name})

  add_executable(${test_target} "${test_src}")
  target_link_libraries(${test_target} ${cub_target})
  cub_clone_target_properties(${test_target} ${cub_target})
  target_include_directories(${test_target} PRIVATE "${CUB_SOURCE_DIR}/test")
  target_compile_definitions(${test_target} PRIVATE CUB_DEBUG_HOST_ASSERTIONS)

  # Add to the active configuration's meta target
  add_dependencies(${config_meta_target} ${test_target})

  # Meta target that builds tests with this name for all configurations:
  if (NOT TARGET ${test_meta_target})
    add_custom_target(${test_meta_target})
  endif()
  add_dependencies(${test_meta_target} ${test_target})

  add_test(NAME ${test_target}
    COMMAND "$<TARGET_FILE:${test_target}>"
  )
endfunction()

# Sets out_var to 1 if the label contains cdp variants, regardless of whether
# or not CDP is enabled in this particular variant.
function(_cub_has_cdp_variant out_var label)
  string(FIND "${label}" "cdp_" idx)
  if (idx EQUAL -1)
    set(${out_var} 0 PARENT_SCOPE)
  else()
    set(${out_var} 1 PARENT_SCOPE)
  endif()
endfunction()

# Sets out_var to 1 if the label contains "cdp_1", e.g. cdp is explicitly
# requested for this variant.
function(_cub_is_cdp_enabled_variant out_var label)
  string(FIND "${label}" "cdp_1" idx)
  if (idx EQUAL -1)
    set(${out_var} 0 PARENT_SCOPE)
  else()
    set(${out_var} 1 PARENT_SCOPE)
  endif()
endfunction()

foreach (test_src IN LISTS test_srcs)
  get_filename_component(test_name "${test_src}" NAME_WE)
  string(REGEX REPLACE "^test_" "" test_name "${test_name}")

  cub_get_test_params("${test_src}" variant_labels variant_defs)
  list(LENGTH variant_labels num_variants)

  # Subtract 1 to support the inclusive endpoint of foreach(...RANGE...):
  math(EXPR range_end "${num_variants} - 1")

  # Verbose output:
  if (num_variants GREATER 0)
    message(VERBOSE "Detected ${num_variants} variants of test '${test_src}':")
    foreach(var_idx RANGE ${range_end})
      math(EXPR i "${var_idx} + 1")
      list(GET variant_labels ${var_idx} label)
      list(GET variant_defs ${var_idx} defs)
      message(VERBOSE "  ${i}: ${test_name} ${label} ${defs}")
    endforeach()
  endif()

  foreach(cub_target IN LISTS CUB_TARGETS)
    cub_get_target_property(config_prefix ${cub_target} PREFIX)

    if (num_variants EQUAL 0)
      # Only one version of this test.
      cub_add_test(test_target ${test_name} "${test_src}" ${cub_target})
      if (CUB_ENABLE_TESTS_WITH_RDC)
        cub_enable_rdc_for_cuda_target(${test_target})
      endif()
    else() # has variants:
      # Meta target to build all parametrizations of the current test for the
      # current CUB_TARGET config
      set(variant_meta_target ${config_prefix}.test.${test_name}.all)
      if (NOT TARGET ${variant_meta_target})
        add_custom_target(${variant_meta_target})
      endif()

      # Meta target to build all parametrizations of the current test for all
      # CUB_TARGET configs
      set(cub_variant_meta_target cub.all.test.${test_name}.all)
      if (NOT TARGET ${cub_variant_meta_target})
        add_custom_target(${cub_variant_meta_target})
      endif()

      # Generate multiple tests, one per variant.
      # See `cub_get_test_params` for details.
      foreach(var_idx RANGE ${range_end})
        list(GET variant_labels ${var_idx} label)
        list(GET variant_defs ${var_idx} defs)
        string(REPLACE ":" ";" defs "${defs}")

        # Check if the test has explicit CDP variants:
        _cub_has_cdp_variant(explicit_cdp "${label}")
        _cub_is_cdp_enabled_variant(enable_cdp "${label}")

        cub_add_test(test_target
          ${test_name}.${label}
          "${test_src}"
          ${cub_target}
        )
        add_dependencies(${variant_meta_target} ${test_target})
        add_dependencies(${cub_variant_meta_target} ${test_target})
        target_compile_definitions(${test_target} PRIVATE ${defs})

        # Enable RDC if the test either:
        # 1. Explicitly requests it (cdp_1 label)
        # 2. Does not have an explicit CDP variant (no cdp_0 or cdp_1) but
        #    RDC testing is globally enabled.
        #
        # Tests that explicitly request no cdp (cdp_0 label) should never enable
        # RDC.
        if (enable_cdp OR ((NOT explicit_cdp) AND CUB_ENABLE_TESTS_WITH_RDC))
          cub_enable_rdc_for_cuda_target(${test_target})
        endif()
      endforeach() # Variant
    endif() # Has variants
  endforeach() # CUB targets
endforeach() # Source file


# Catch2 tests
find_package(CUDAToolkit REQUIRED)
foreach(cub_target IN LISTS CUB_TARGETS)
  cub_get_target_property(config_prefix ${cub_target} PREFIX)

  # The actual name of the test's target:
  set(target ${config_prefix}.test.catch2_helper)

  add_library(${target} SHARED
    c2h/custom_type.cuh 
    c2h/generators.cuh 
    c2h/generators.cu)

  target_link_libraries(${target} PRIVATE CUDA::curand)

  set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)

  if ("NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
    target_link_options(${target} PRIVATE "-shared")
    target_link_options(${target} PRIVATE "-cuda")
    target_compile_options(${target} PRIVATE "-fPIC")
  endif()

  target_include_directories(${target} 
    PUBLIC "${CUB_SOURCE_DIR}/test")

  cub_clone_target_properties(${target} ${cub_target})
endforeach()

## cub_add_catch2_test 
#
# Add a catch2 test executable and register it with ctest.
#
# src: The source file that implements the test.
# name: The name of the test minus "<config_prefix>.catch2_test_." For example,
#   test/catch2_test_block_load.cu will be "block_load".
# defs: Defines to be added to the test.
#
function(cub_add_catch2_test src name defs target config_prefix)
  set(name ${config_prefix}.test.${name})

  if (CUB_SEPARATE_CATCH2)
    add_executable(${name} "${src}")
    add_test(NAME ${name} COMMAND "$<TARGET_FILE:${name}>")
    set(${name} cub.all.test.${name})
    list(APPEND defs "CUB_CONFIG_MAIN")
  else ()
    add_library(${name} OBJECT "${src}")
  endif()

  target_link_libraries(${name} PRIVATE ${target} Metal Catch2::Catch2 ${config_prefix}.test.catch2_helper)
  set_property(TARGET ${name} PROPERTY POSITION_INDEPENDENT_CODE ON)

  if ("NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
    target_link_options(${name} PRIVATE "-cuda")
    target_compile_options(${name} PRIVATE "-fPIC")
  endif ()

  cub_clone_target_properties(${name} ${target})
  target_include_directories(${name} PRIVATE "${CUB_SOURCE_DIR}/test")
  target_compile_definitions(${name} PRIVATE ${defs})
endfunction()

foreach(cub_target IN LISTS CUB_TARGETS)
  cub_get_target_property(config_prefix ${cub_target} PREFIX)

  # The actual name of the test's target:
  set(test_meta_target cub.all.test.${test_name})
  set(test_prefix_target ${config_prefix}.tests)

  if (NOT CUB_SEPARATE_CATCH2)
    set(test_target ${config_prefix}.catch2_test)
    add_executable(${test_target} catch2_runner.cu)
    set(${test_target} cub.all.test.${test_target})
  endif()

  foreach (test_src IN LISTS catch2_test_srcs)
    get_filename_component(test_name "${test_src}" NAME_WE)
    string(REGEX REPLACE "^catch2_test_" "" test_name "${test_name}")

    cub_get_test_params("${test_src}" variant_labels variant_defs)
    list(LENGTH variant_labels num_variants)

    # Subtract 1 to support the inclusive endpoint of foreach(...RANGE...):
    math(EXPR range_end "${num_variants} - 1")

    # Verbose output:
    if (NOT num_variants EQUAL 0)
      message(VERBOSE "Detected ${num_variants} variants of test '${test_src}':")
      foreach(var_idx RANGE ${range_end})
        math(EXPR i "${var_idx} + 1")
        list(GET variant_labels ${var_idx} label)
        list(GET variant_defs ${var_idx} defs)
        message(VERBOSE "  ${i}: ${test_name} ${label} ${defs}")
      endforeach()
    endif()

    cub_get_target_property(config_prefix ${cub_target} PREFIX)

    if (num_variants EQUAL 0)
      # Only one version of this test.
      cub_add_catch2_test(
        "${test_src}"
        ${test_name} 
        "VAR_IDX=0" 
        ${cub_target} 
        ${config_prefix})

      if (NOT CUB_SEPARATE_CATCH2)
        target_link_libraries(${test_target} PRIVATE ${config_prefix}.test.${test_name})
      endif()
    else()
      # Meta target to build all parametrizations of the current test for the
      # current CUB_TARGET config
      set(variant_meta_target ${config_prefix}.test.${test_name}.all)
      add_custom_target(${variant_meta_target})

      # Meta target to build all parametrizations of the current test for all
      # CUB_TARGET configs
      set(cub_variant_meta_target cub.all.test.${test_name}.all)
      if (NOT TARGET ${cub_variant_meta_target})
        add_custom_target(${cub_variant_meta_target})
      endif()

      # Generate multiple tests, one per variant.
      # See `cub_get_test_params` for details.
      foreach(var_idx RANGE ${range_end})
        list(GET variant_labels ${var_idx} label)
        list(GET variant_defs ${var_idx} defs)
        string(REPLACE ":" ";" defs "${defs}")
        set(defs "${defs}; VAR_IDX=${var_idx}")

        cub_add_catch2_test(
          "${test_src}"
          ${test_name}.${label} 
          "${defs}" 
          ${cub_target}
          ${config_prefix})

        if (NOT CUB_SEPARATE_CATCH2)
          target_link_libraries(${test_target} PRIVATE ${config_prefix}.test.${test_name}.${label})
        else()
          add_dependencies(${cub_variant_meta_target} ${test_target})
        endif()
      endforeach()
    endif()
  endforeach()

  if (NOT CUB_SEPARATE_CATCH2)
    target_link_libraries(${test_target} PRIVATE
            ${cub_target}
            Metal
            Catch2::Catch2)

    if ("NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
      target_link_options(${test_target} PRIVATE "-cuda")
    endif ()

    cub_clone_target_properties(${test_target} ${cub_target})
    target_include_directories(${test_target} PRIVATE "${CUB_SOURCE_DIR}/test")

    # Add to the active configuration's meta target
    add_dependencies(${config_meta_target} ${test_target})

    # Meta target that builds tests with this name for all configurations:
    if (NOT TARGET ${test_meta_target})
      add_custom_target(${test_meta_target})
    endif()
    add_dependencies(${test_meta_target} ${test_target})
    add_dependencies(${test_prefix_target} ${test_target})

    add_test(NAME ${test_target} COMMAND "$<TARGET_FILE:${test_target}>")
  endif()
endforeach()

add_subdirectory(cmake)
