# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(nccl_srdsim LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SRDSIM_BUILD_TESTS "Build the host tests (no GPU, no NCCL needed)" ON)
option(SRDSIM_BUILD_PLUGIN "Build libnccl-rma-srdsim.so (needs the NCCL plugin headers)" OFF)
option(SRDSIM_WITH_CUDA "Plugin: deliver into device windows through the CUDA driver API" ON)
option(SRDSIM_WITH_GDRCOPY "Plugin: deliver into device windows through GDRCopy when available" OFF)
set(SRDSIM_SANITIZER "" CACHE STRING
    "Plugin and tests: build with a sanitizer (address, undefined or thread; empty for none)")
set_property(CACHE SRDSIM_SANITIZER PROPERTY STRINGS "" address undefined thread)
set(NCCL_PLUGIN_INCLUDE_DIRS "" CACHE STRING
    "Directories holding nccl.h and the plugin headers (nccl_rma.h), separated by ;")

set(SRDSIM_WARNINGS -Wall -Wextra)
if(SRDSIM_SANITIZER)
  if(NOT SRDSIM_SANITIZER MATCHES "^(address|undefined|thread)$")
    message(FATAL_ERROR "SRDSIM_SANITIZER must be address, undefined or thread, not '${SRDSIM_SANITIZER}'")
  endif()
  # UBSan stops at a finding only when built not to recover; ASan always stops; ThreadSanitizer stops only
  # through TSAN_OPTIONS, which the tests below are given.
  add_compile_options(-fsanitize=${SRDSIM_SANITIZER} -fno-sanitize-recover=all -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${SRDSIM_SANITIZER})
endif()

find_package(Threads REQUIRED)

add_library(srdsim_core INTERFACE)
target_include_directories(srdsim_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(SRDSIM_BUILD_TESTS)
  enable_testing()
  set(SRDSIM_HOST_TESTS
      test_srdsim_schedule test_srdsim_transport test_srdsim_get test_srdsim_adversarial
      test_srdsim_liveness test_srdsim_async test_srdsim_mapping
      model_srdsim_schedule_exhaustive model_srdsim_framing)
  foreach(t IN LISTS SRDSIM_HOST_TESTS)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE srdsim_core Threads::Threads)
    target_compile_options(${t} PRIVATE ${SRDSIM_WARNINGS})
    add_test(NAME ${t} COMMAND ${t})
    if(SRDSIM_SANITIZER)
      # The same options locally as in CI, so that a finding fails the test wherever it is run.
      set_tests_properties(${t} PROPERTIES ENVIRONMENT
        "ASAN_OPTIONS=detect_leaks=1:abort_on_error=1;UBSAN_OPTIONS=print_stacktrace=1;TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1")
    endif()
  endforeach()
endif()

if(SRDSIM_BUILD_PLUGIN)
  if(NOT NCCL_PLUGIN_INCLUDE_DIRS)
    message(FATAL_ERROR "set NCCL_PLUGIN_INCLUDE_DIRS to the directories holding nccl.h and nccl_rma.h")
  endif()
  if(SRDSIM_WITH_GDRCOPY AND NOT SRDSIM_WITH_CUDA)
    message(FATAL_ERROR "SRDSIM_WITH_GDRCOPY needs SRDSIM_WITH_CUDA")
  endif()
  add_library(nccl-rma-srdsim SHARED src/nccl_rma_srdsim.cc)
  target_include_directories(nccl-rma-srdsim SYSTEM PRIVATE ${NCCL_PLUGIN_INCLUDE_DIRS})
  target_link_libraries(nccl-rma-srdsim PRIVATE srdsim_core Threads::Threads)
  # -g1 keeps line tables, so an address from a crash report resolves with addr2line.
  target_compile_options(nccl-rma-srdsim PRIVATE ${SRDSIM_WARNINGS} -g1)
  # nccl.h includes cuda_runtime.h, so the CUDA headers are needed in every configuration. Only the
  # CUDA delivery path links the driver API and NVRTC.
  find_package(CUDAToolkit REQUIRED)
  target_include_directories(nccl-rma-srdsim SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  if(SRDSIM_WITH_CUDA)
    target_compile_definitions(nccl-rma-srdsim PRIVATE SRDSIM_WITH_CUDA)
    target_link_libraries(nccl-rma-srdsim PRIVATE CUDA::cuda_driver CUDA::nvrtc)
  endif()
  if(SRDSIM_WITH_GDRCOPY)
    find_path(GDRAPI_INCLUDE_DIR gdrapi.h REQUIRED)
    find_library(GDRAPI_LIB gdrapi REQUIRED)
    target_include_directories(nccl-rma-srdsim SYSTEM PRIVATE ${GDRAPI_INCLUDE_DIR})
    target_compile_definitions(nccl-rma-srdsim PRIVATE SRDSIM_WITH_GDRCOPY)
    target_link_libraries(nccl-rma-srdsim PRIVATE ${GDRAPI_LIB})
  endif()
  include(GNUInstallDirs)
  install(TARGETS nccl-rma-srdsim LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
