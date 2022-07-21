# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

# Meta target for all configs' header builds:
add_custom_target(cub.all.headers)

file(GLOB_RECURSE headers
  RELATIVE "${CUB_SOURCE_DIR}/cub"
  CONFIGURE_DEPENDS
  cub/*.cuh
)

set(headertest_srcs)
foreach (header IN LISTS headers)
  set(headertest_src "headers/${header}.cu")
  configure_file("${CUB_SOURCE_DIR}/cmake/header_test.in" "${headertest_src}")
  list(APPEND headertest_srcs "${headertest_src}")
endforeach()

function(cub_add_header_test target_name srcs cub_target)
  add_library(${target_name} OBJECT ${srcs})
  target_link_libraries(${target_name} PUBLIC ${cub_target})

  # Wrap Thrust/CUB in a custom namespace to check proper use of ns macros:
  target_compile_definitions(${target_name} PRIVATE
    "THRUST_WRAPPED_NAMESPACE=wrapped_thrust"
    "CUB_WRAPPED_NAMESPACE=wrapped_cub"
    )
  cub_clone_target_properties(${target_name} ${cub_target})
endfunction()

foreach(cub_target IN LISTS CUB_TARGETS)
  cub_get_target_property(config_prefix ${cub_target} PREFIX)

  set(headertest_target ${config_prefix}.headers) # Metatarget
  add_custom_target(${headertest_target})
  add_dependencies(cub.all.headers ${headertest_target})
  add_dependencies(${config_prefix}.all ${headertest_target})

  set(headertest_rdc_target ${headertest_target}.rdc)
  cub_add_header_test(
    ${headertest_rdc_target}
    "${headertest_srcs}"
    ${cub_target}
  )
  cub_set_rdc_state(${headertest_rdc_target} ON)
  add_dependencies(${headertest_target} ${headertest_rdc_target})

  set(headertest_no_rdc_target ${headertest_target}.no_rdc)
  cub_add_header_test(
    ${headertest_no_rdc_target}
    "${headertest_srcs}"
    ${cub_target}
  )
  cub_set_rdc_state(${headertest_no_rdc_target} OFF)
  add_dependencies(${headertest_target} ${headertest_no_rdc_target})

  set(headertest_no_bf16_target ${headertest_target}.no_bf16)
  cub_add_header_test(
    ${headertest_no_bf16_target}
    "${headertest_srcs}"
    ${cub_target}
  )
  cub_set_rdc_state(${headertest_no_bf16_target} OFF)
  target_compile_definitions(${headertest_no_bf16_target}
    PRIVATE "CUB_DISABLE_BF16_SUPPORT"
  )
  add_dependencies(${headertest_target} ${headertest_no_bf16_target})
endforeach()
