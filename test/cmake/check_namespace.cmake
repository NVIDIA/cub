# Check all files in thrust to make sure that they use
# CUB_NAMESPACE_BEGIN/END instead of bare `namespace cub {}` declarations.
#
# This is run as a ctest test named `cub.test.cmake.check_namespace`, or
# manually with:
# cmake -D "CUB_SOURCE_DIR=<CUB project root>" -P check_namespace.cmake

cmake_minimum_required(VERSION 3.15)

set(exclusions
  # This defines the macros and must have bare namespace declarations:
  cub/util_namespace.cuh
)

function(count_substrings input search_regex output_var)
  string(REGEX MATCHALL "${search_regex}" matches "${input}")
  list(LENGTH matches num_matches)
  set(${output_var} ${num_matches} PARENT_SCOPE)
endfunction()

set(bare_ns_regex "namespace[ \n\r\t]+cub[ \n\r\t]*\\{")

# Validation check for the above regex:
count_substrings([=[
namespace cub{
namespace cub {
namespace  cub  {
 namespace cub {
namespace cub
{
namespace
cub
{
]=]
  ${bare_ns_regex} valid_count)
if (NOT valid_count EQUAL 6)
  message(FATAL_ERROR "Validation of bare namespace regex failed: "
                      "Matched ${valid_count} times, expected 6.")
endif()

set(found_errors 0)
file(GLOB_RECURSE cub_srcs
  RELATIVE "${CUB_SOURCE_DIR}"
  "${CUB_SOURCE_DIR}/*.cuh"
  "${CUB_SOURCE_DIR}/*.cu"
  "${CUB_SOURCE_DIR}/*.h"
  "${CUB_SOURCE_DIR}/*.cpp"
)

foreach(src ${cub_srcs})
  if (${src} IN_LIST exclusions)
    continue()
  endif()

  file(READ "${CUB_SOURCE_DIR}/${src}" src_contents)

  count_substrings("${src_contents}" "${bare_ns_regex}" bare_ns_count)
  count_substrings("${src_contents}" CUB_NS_PREFIX prefix_count)
  count_substrings("${src_contents}" CUB_NS_POSTFIX postfix_count)
  count_substrings("${src_contents}" CUB_NAMESPACE_BEGIN begin_count)
  count_substrings("${src_contents}" CUB_NAMESPACE_END end_count)

  if (NOT bare_ns_count EQUAL 0)
    message("'${src}' contains 'namespace cub {...}'. Replace with CUB_NAMESPACE macros.")
    set(found_errors 1)
  endif()

  if (NOT prefix_count EQUAL 0)
    message("'${src}' contains 'CUB_NS_PREFIX'. Replace with CUB_NAMESPACE macros.")
    set(found_errors 1)
  endif()

  if (NOT postfix_count EQUAL 0)
    message("'${src}' contains 'CUB_NS_POSTFIX'. Replace with CUB_NAMESPACE macros.")
    set(found_errors 1)
  endif()

  if (NOT begin_count EQUAL end_count)
    message("'${src}' namespace macros are unbalanced:")
    message(" - CUB_NAMESPACE_BEGIN occurs ${begin_count} times.")
    message(" - CUB_NAMESPACE_END   occurs ${end_count} times.")
    set(found_errors 1)
  endif()

endforeach()

if (NOT found_errors EQUAL 0)
  message(FATAL_ERROR "Errors detected.")
endif()
