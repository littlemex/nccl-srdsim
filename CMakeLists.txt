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
option(SRDSIM_BUILD_FI_TEST "Build the libfabric provider's test (needs libfabric and SRDSIM_FI_PROVIDER)" OFF)
set(SRDSIM_FI_PROVIDER "" CACHE FILEPATH "The libsrdsim-fi.so the libfabric provider's test loads")
set(SRDSIM_SANITIZER "" CACHE STRING
    "Plugin and tests: build with a sanitizer (address, undefined or thread; empty for none)")
set_property(CACHE SRDSIM_SANITIZER PROPERTY STRINGS "" address undefined thread)
set(NCCL_PLUGIN_INCLUDE_DIRS "" CACHE STRING
    "Directories holding nccl.h and the plugin headers (nccl_rma.h), separated by ;")

set(SRDSIM_WARNINGS -Wall -Wextra)
if(SRDSIM_ASAN)
  message(FATAL_ERROR "SRDSIM_ASAN was replaced by -DSRDSIM_SANITIZER=address (clear a cached value with -USRDSIM_ASAN)")
endif()
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
        "ASAN_OPTIONS=abort_on_error=1;UBSAN_OPTIONS=print_stacktrace=1;TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1"
        TIMEOUT 300)
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

  # The plugin harness (tests/plugin): drives the built plugin through its op table as NCCL does, once
  # per table version listed here. Host windows only, so it needs no GPU.
  if(SRDSIM_BUILD_TESTS)
    set(SRDSIM_HARNESS_TABLES "14;15;16" CACHE STRING "Op-table versions the plugin harness runs against")
    add_executable(test_plugin tests/plugin/test_plugin.cc)
    target_include_directories(test_plugin SYSTEM PRIVATE ${NCCL_PLUGIN_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
    target_include_directories(test_plugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_plugin PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
    if(SRDSIM_WITH_CUDA)
      # The device-window cases; they skip themselves on a host without a GPU.
      target_compile_definitions(test_plugin PRIVATE HARNESS_CUDA)
      target_link_libraries(test_plugin PRIVATE CUDA::cuda_driver)
    endif()
    target_compile_options(test_plugin PRIVATE ${SRDSIM_WARNINGS})
    foreach(v IN LISTS SRDSIM_HARNESS_TABLES)
      add_test(NAME plugin_harness_v${v} COMMAND test_plugin $<TARGET_FILE:nccl-rma-srdsim> ${v})
      set_tests_properties(plugin_harness_v${v} PROPERTIES TIMEOUT 900)
      if(SRDSIM_SANITIZER)
        set_tests_properties(plugin_harness_v${v} PROPERTIES ENVIRONMENT
          "ASAN_OPTIONS=abort_on_error=1;UBSAN_OPTIONS=print_stacktrace=1;TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1")
      endif()
    endforeach()
  endif()
endif()

# The libfabric provider is built inside a libfabric source tree by scripts/build_fi_provider.sh; this
# builds and registers its test, against that libfabric.
if(SRDSIM_BUILD_FI_TEST)
  if(NOT SRDSIM_FI_PROVIDER)
    message(FATAL_ERROR "set SRDSIM_FI_PROVIDER to the libsrdsim-fi.so that scripts/build_fi_provider.sh built")
  endif()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBFABRIC REQUIRED IMPORTED_TARGET libfabric)
  add_executable(test_fi_provider tests/fi/test_fi_provider.cc)
  target_link_libraries(test_fi_provider PRIVATE PkgConfig::LIBFABRIC)
  target_compile_options(test_fi_provider PRIVATE ${SRDSIM_WARNINGS})
  enable_testing()
  add_test(NAME fi_provider COMMAND test_fi_provider ${SRDSIM_FI_PROVIDER})
  set_tests_properties(fi_provider PROPERTIES TIMEOUT 120)
  add_executable(test_fi_transfer tests/fi/test_fi_transfer.cc)
  target_link_libraries(test_fi_transfer PRIVATE PkgConfig::LIBFABRIC)
  target_compile_options(test_fi_transfer PRIVATE ${SRDSIM_WARNINGS})
  add_executable(test_fi_wire tests/fi/test_fi_wire.cc)
  target_link_libraries(test_fi_wire PRIVATE PkgConfig::LIBFABRIC)
  target_compile_options(test_fi_wire PRIVATE ${SRDSIM_WARNINGS})
  add_test(NAME fi_wire COMMAND test_fi_wire ${SRDSIM_FI_PROVIDER} 100)
  set_tests_properties(fi_wire PROPERTIES TIMEOUT 600)
  add_executable(test_fi_multipeer tests/fi/test_fi_multipeer.cc)
  target_link_libraries(test_fi_multipeer PRIVATE PkgConfig::LIBFABRIC)
  target_compile_options(test_fi_multipeer PRIVATE ${SRDSIM_WARNINGS})
  add_test(NAME fi_multipeer COMMAND test_fi_multipeer ${SRDSIM_FI_PROVIDER})
  set_tests_properties(fi_multipeer PROPERTIES TIMEOUT 180)
  add_executable(test_fi_cq_threads tests/fi/test_fi_cq_threads.cc)
  target_link_libraries(test_fi_cq_threads PRIVATE PkgConfig::LIBFABRIC Threads::Threads)
  target_compile_options(test_fi_cq_threads PRIVATE ${SRDSIM_WARNINGS})
  # Both threading models libfabric disarms the completion queue's lock for.
  foreach(m IN ITEMS completion domain)
    add_test(NAME fi_cq_threads_${m} COMMAND test_fi_cq_threads ${SRDSIM_FI_PROVIDER} ${m})
    set_tests_properties(fi_cq_threads_${m} PROPERTIES TIMEOUT 180)
  endforeach()
  # The negative control for the guard above is a CI step, not a test here: ctest's WILL_FAIL passes on any
  # non-zero exit, so it would also go green if the provider failed to load. The step in .github/workflows/ci.yml
  # checks that the run fails with ThreadSanitizer's own exit code and that the report names the completion queue.
  foreach(s IN ITEMS sendrecv tagged directsend closeinject closewrite inject unreachable truncate read badkey writedata reorder)
    add_test(NAME fi_transfer_${s} COMMAND test_fi_transfer ${SRDSIM_FI_PROVIDER} ${s})
    set_tests_properties(fi_transfer_${s} PROPERTIES TIMEOUT 180)
  endforeach()
  # The same reordering never shows when everything is released in submission order.
  add_test(NAME fi_transfer_reorder_inorder COMMAND test_fi_transfer ${SRDSIM_FI_PROVIDER} reorder)
  set_tests_properties(fi_transfer_reorder_inorder PROPERTIES TIMEOUT 180 ENVIRONMENT SRDSIM_MODE=inorder)
  # Negative control: a remote completion raised before its write has landed must be caught.
  add_test(NAME fi_transfer_writedata_early COMMAND test_fi_transfer ${SRDSIM_FI_PROVIDER} writedata)
  set_tests_properties(fi_transfer_writedata_early PROPERTIES TIMEOUT 180 ENVIRONMENT SRDSIM_FI_CQ_DATA_EARLY=1
                       WILL_FAIL TRUE)
endif()
