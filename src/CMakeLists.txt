cmake_minimum_required(VERSION 3.18)
project(tacker VERSION 0.4.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(TACKER_BUILD_LEGACY "Build the original CUDA benchmark executable" ON)
option(TACKER_ENABLE_CUDA_BACKEND "Build the CUDA runtime backend when CUDA is available" ON)
option(TACKER_BUILD_TESTS "Build CPU-only runtime tests" ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TACKER_RUNTIME_SOURCES
    runtime/Kernel.cc
    runtime/TaskGraph.cc
    runtime/Registry.cc
    runtime/Scheduler.cc)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(TACKER_CUDA_AVAILABLE ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()
    find_package(CUDAToolkit REQUIRED)
else()
    set(TACKER_CUDA_AVAILABLE OFF)
endif()

if(TACKER_CUDA_AVAILABLE AND TACKER_ENABLE_CUDA_BACKEND)
    list(APPEND TACKER_RUNTIME_SOURCES runtime/CudaExecutionBackend.cc)
endif()

add_library(tacker_runtime SHARED ${TACKER_RUNTIME_SOURCES})
add_library(Tacker::runtime ALIAS tacker_runtime)
set_target_properties(tacker_runtime PROPERTIES
    EXPORT_NAME runtime
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
target_include_directories(tacker_runtime
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>)
target_compile_features(tacker_runtime PUBLIC cxx_std_11)

if(TACKER_CUDA_AVAILABLE AND TACKER_ENABLE_CUDA_BACKEND)
    target_link_libraries(tacker_runtime PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_compile_definitions(tacker_runtime PUBLIC TACKER_RUNTIME_HAS_CUDA=1)
else()
    message(STATUS "CUDA runtime backend disabled; building CPU-only libtacker_runtime")
endif()

if(TACKER_BUILD_TESTS)
    enable_testing()
    add_executable(tacker_runtime_tests ../tests/runtime/runtime_tests.cc)
    target_link_libraries(tacker_runtime_tests PRIVATE tacker_runtime)
    add_test(NAME tacker_runtime_tests COMMAND tacker_runtime_tests)
endif()

# Make the runtime consumable without pulling in the original benchmark.
# Public headers intentionally retain their `runtime/...` include prefix.
install(TARGETS tacker_runtime
    EXPORT TackerRuntimeTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY runtime/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/runtime
    FILES_MATCHING PATTERN "*.h")

configure_package_config_file(
    cmake/TackerRuntimeConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/TackerRuntimeConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TackerRuntime)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/TackerRuntimeConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TackerRuntimeConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TackerRuntimeConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TackerRuntime)
install(EXPORT TackerRuntimeTargets
    FILE TackerRuntimeTargets.cmake
    NAMESPACE Tacker::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TackerRuntime)

# Keep the original benchmark available when a CUDA toolkit is installed. The
# generic runtime above deliberately does not depend on this legacy target.
if(TACKER_BUILD_LEGACY AND TACKER_CUDA_AVAILABLE)
    set(LOG_FILE_PATH ${PROJECT_SOURCE_DIR}/log/${PROJECT_VERSION}.log)
    string(TIMESTAMP COMPILE_TIME %Y%m%d-%H%M%S)
    configure_file(TackerConfig.h.in TackerConfig.h)

    find_path(CUDNN_INCLUDE_DIR cudnn.h
        HINTS ${CUDAToolkit_ROOT}
        PATH_SUFFIXES cuda/include include)
    find_library(CUDNN_LIBRARY cudnn
        HINTS ${CUDAToolkit_ROOT}
        PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

    set(TACKER_LEGACY_SOURCES
        main.cu Logger.cc TaskManager.cu Task.cu ModuleCenter.cu readatom.c
        Recorder.cc json.cc Creator.cu
        dnn/inception3/inception3.cu dnn/resnet50/resnet50.cu
        dnn/bert/bert.cu dnn/vgg11/vgg11.cu dnn/vgg16/vgg16.cu dnn/vit/vit.cu)

    add_executable(tacker ${TACKER_LEGACY_SOURCES})
    set_target_properties(tacker PROPERTIES
        CUDA_ARCHITECTURES 86
        CUDA_STANDARD 11
        CUDA_STANDARD_REQUIRED ON)
    target_compile_options(tacker PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xptxas=-dlcm=ca;-w>")
    target_include_directories(tacker PUBLIC
        ${PROJECT_BINARY_DIR}
        ${PROJECT_SOURCE_DIR}
        ${CUDNN_INCLUDE_DIR})
    target_link_libraries(tacker
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cublas
        CUDA::curand
        ${CUDNN_LIBRARY})
elseif(TACKER_BUILD_LEGACY)
    message(STATUS "CUDA toolkit not found; original tacker executable will not be built")
endif()
