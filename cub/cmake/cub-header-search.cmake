unset(_CUB_VERSION_INCLUDE_DIR CACHE) # Clear old result to force search
find_path(_CUB_VERSION_INCLUDE_DIR cub/version.cuh
  NO_DEFAULT_PATH # Only search explicit paths below:
  PATHS
    "${CMAKE_CURRENT_LIST_DIR}/../.."            # Source tree
)
set_property(CACHE _CUB_VERSION_INCLUDE_DIR PROPERTY TYPE INTERNAL)
