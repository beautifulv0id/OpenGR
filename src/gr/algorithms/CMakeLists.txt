project (OpenGR-AlgoLib)

set(algo_ROOT "${SRC_DIR}/gr/algorithms")

file(GLOB_RECURSE algo_relative_INCLUDE ${algo_ROOT}/*.h ${algo_ROOT}/*.hpp)

# compute accel_INCLUDE to get absolute filenames
set(algo_INCLUDE "")
set(algo_installed_INCLUDE "")
foreach(f ${algo_relative_INCLUDE})
    list(APPEND algo_INCLUDE "${f}")
    file(RELATIVE_PATH f_relative "${SRC_DIR}" "${f}")
    list(APPEND algo_installed_INCLUDE "include/${f_relative}")
endforeach(f)

add_library(algo INTERFACE)
target_sources(algo INTERFACE
    $<BUILD_INTERFACE:${algo_INCLUDE}>
    $<INSTALL_INTERFACE:${algo_installed_INCLUDE}>
    )
add_dependencies(algo gr::accel gr::utils)
set_target_properties(algo PROPERTIES
  INTERFACE_COMPILE_FEATURES cxx_std_17
)

if(OpenGR_USE_CHEALPIX)
    include_directories(${Chealpix_INCLUDE_DIR})
    add_dependencies(algo chealpix)
endif(OpenGR_USE_CHEALPIX)

set(targets_export_name "${PROJECT_NAME}Targets")
install( TARGETS algo
    EXPORT "${targets_export_name}"
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include  )

add_library(gr::algo ALIAS algo)

#############################################
#
# HACK: have the files showing in the IDE, under the name 'opengr_algo_IDE'
# Source: https://stackoverflow.com/questions/27039019/how-to-have-cmake-show-headers-that-are-not-part-of-any-binary-target-in-the-ide/29214327#29214327
#
add_custom_target(opengr_algo_IDE SOURCES ${algo_INCLUDE})
