project(wavelet)
cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

message(STATUS "Configuring wavelet library")

option(WAVELET_BUILD_TESTS "Build the test executables" OFF)
option(WAVELET_ENABLE_VECTOR "Build the library with vector encoders and decoders" OFF)
option(WAVELET_ENABLE_MATRIX "Build the library with matrix encoders and decoders" OFF)
option(WAVELET_ENABLE_ARRAY_METADATA "Build the library with array metadata support" OFF)
option(WAVELET_ENABLE_SLICE_METADATA "Build the library with slice metadata support" OFF)

# output directory of the crate
set(BDIR ${CMAKE_BINARY_DIR}/libwavelet)

# build with ffi feature
set(FEATURES "ffi")
set(FEATURES_DEF "WAVELET_RS_FEAT_FFI")

# enable optional features
if(WAVELET_ENABLE_VECTOR)
    string(APPEND FEATURES ",ffi_vec")
    list(APPEND FEATURES_DEF "WAVELET_RS_FEAT_FFI_VEC")
endif(WAVELET_ENABLE_VECTOR)

if(WAVELET_ENABLE_MATRIX)
    string(APPEND FEATURES ",ffi_mat")
    list(APPEND FEATURES_DEF "WAVELET_RS_FEAT_FFI_MAT")
endif(WAVELET_ENABLE_MATRIX)

if(WAVELET_ENABLE_ARRAY_METADATA)
    string(APPEND FEATURES ",ffi_metadata_arr")
    list(APPEND FEATURES_DEF "WAVELET_RS_FEAT_FFI_MEATADATA_ARR")
endif(WAVELET_ENABLE_ARRAY_METADATA)

if(WAVELET_ENABLE_SLICE_METADATA)
    string(APPEND FEATURES ",ffi_metadata_slice")
    list(APPEND FEATURES_DEF "WAVELET_RS_FEAT_FFI_MEATADATA_SLICE")
endif(WAVELET_ENABLE_SLICE_METADATA)

# enable ExternalProject CMake module for building the rust crate
include(ExternalProject)

# add external project building the crate
ExternalProject_Add(
    ext_libwavelet
    BUILD_ALWAYS         ON
    CONFIGURE_COMMAND    ""
    BUILD_COMMAND        cargo build --target-dir ${BDIR} --features ${FEATURES} "$<IF:$<CONFIG:Release>,--release,>"
    BUILD_BYPRODUCTS     ${BDIR}/$<IF:$<CONFIG:Release>,release,debug>/libwavelet_rs.a
    INSTALL_COMMAND      ""
    SOURCE_DIR           "${CMAKE_CURRENT_SOURCE_DIR}"
    BINARY_DIR           "${CMAKE_CURRENT_SOURCE_DIR}"
    PREFIX               "libwavelet"
    LOG_DIR              "log"
    LOG_BUILD             ON
    LOG_OUTPUT_ON_FAILURE ON
)

# add pseudo target to the imported library
add_library(libwavelet STATIC IMPORTED GLOBAL)
add_dependencies(libwavelet ext_libwavelet)

set_target_properties(libwavelet PROPERTIES
    IMPORTED_CONFIGURATIONS "Debug;Release"
    IMPORTED_LOCATION       "${BDIR}/release/libwavelet_rs.a"
    IMPORTED_LOCATION_DEBUG "${BDIR}/debug/libwavelet_rs.a"
)
target_compile_definitions(libwavelet INTERFACE ${FEATURES_DEF})
target_include_directories(libwavelet INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(BUILD_TESTS)
    add_subdirectory(tests)
endif(BUILD_TESTS)
