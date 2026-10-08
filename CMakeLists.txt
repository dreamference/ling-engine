cmake_minimum_required(VERSION 3.28)
project(ling_engine LANGUAGES CXX CUDA VERSION 0.1.0)

# ling-engine v0: one model (Qwen3.8-27B, NVFP4 MLP + FP8 attention), one GPU (GB10, sm_121).
# SPEC.md §8: C++20 and CUDA throughout; separate targets for the core, the tokenizer and the server,
# so a server change never recompiles a kernel.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# sm_121a, not sm_121: the FP4 conversions and block-scaled MMAs are arch-specific (reports/M0.md §2).
set(CMAKE_CUDA_ARCHITECTURES 121a)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Debug builds compile every kernel with device debug info (SPEC.md §8, "Debuggable from day one").
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo")
string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr")

option(LING_SANITIZE "Build host code with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(LING_SANITIZE)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json 3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)
find_package(ICU REQUIRED COMPONENTS uc)

# The engine core: weights, kernels, the forward pass and sampling.
add_library(ling_core STATIC
  src/core/safetensors.cpp
  src/core/kernels.cu
  src/core/stream_gemm.cu
  src/core/topk.cu
  src/core/draft_kernels.cu
  src/core/drafter.cpp
  src/core/speculate.cpp
  src/core/model.cu
  src/core/engine.cpp
)
target_include_directories(ling_core PUBLIC src)
target_link_libraries(ling_core PUBLIC CUDA::cudart CUDA::cublas CUDA::nvtx3 nlohmann_json::nlohmann_json)
set_target_properties(ling_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# The tokenizer: byte-level BPE with Qwen's pre-tokenizer pattern (PCRE2) and NFC (ICU).
add_library(ling_tokenizer STATIC src/tokenizer/tokenizer.cpp src/tokenizer/chat_template.cpp)
target_include_directories(ling_tokenizer PUBLIC src)
target_link_libraries(ling_tokenizer PUBLIC PkgConfig::PCRE2 ICU::uc nlohmann_json::nlohmann_json)

# ling-run: token ids or text in, tokens out; the correctness and speed harness.
add_executable(ling-run tools/ling_run.cpp)
target_link_libraries(ling-run PRIVATE ling_core ling_tokenizer)

# ling-tokenize: prints token ids for each input line (tested against the reference tokenizer).
add_executable(ling-tokenize tools/ling_tokenize.cpp)
target_link_libraries(ling-tokenize PRIVATE ling_tokenizer)
add_executable(ling-template tools/ling_template.cpp)
target_link_libraries(ling-template PRIVATE ling_tokenizer)

# The API layer without the HTTP library, so it is testable on its own.
add_library(ling_api STATIC src/serve/api.cpp)
target_link_libraries(ling_api PUBLIC ling_core ling_tokenizer)

# ling-serve: the HTTP server (proxygen), built when proxygen is found.
set(LING_PROXYGEN_PREFIX "$ENV{HOME}/ling-deps/proxygen" CACHE PATH "Where proxygen and its dependencies are installed")
# getdeps installs each project under its own directory of the prefix (folly/, fizz/, proxygen/, ...).
file(GLOB _ling_dep_prefixes "${LING_PROXYGEN_PREFIX}/*")
list(APPEND CMAKE_PREFIX_PATH ${_ling_dep_prefixes})
# proxygen's dependency configs look for Sodium, Zstd and others through fbcode_builder's Find modules.
set(LING_FBCODE_CMAKE "$ENV{HOME}/ling-deps/src/proxygen/build/fbcode_builder/CMake" CACHE PATH
    "fbcode_builder's CMake modules (from the proxygen checkout used to build the dependencies)")
list(APPEND CMAKE_MODULE_PATH "${LING_FBCODE_CMAKE}")
find_package(proxygen CONFIG QUIET)
if(proxygen_FOUND)
  add_executable(ling-serve src/serve/main.cpp)
  target_link_libraries(ling-serve PRIVATE ling_api proxygen::proxygenhttpserver Folly::folly_init_init)
else()
  message(WARNING "proxygen not found under ${LING_PROXYGEN_PREFIX}: ling-serve is not built")
endif()

enable_testing()
add_executable(ling-kernel-tests tests/kernel_tests.cu)
target_link_libraries(ling-kernel-tests PRIVATE ling_core)
add_test(NAME kernels COMMAND ling-kernel-tests)
add_executable(ling-stream-tests tests/stream_tests.cu)
target_link_libraries(ling-stream-tests PRIVATE ling_core)
add_test(NAME stream COMMAND ling-stream-tests)
add_executable(ling-api-tests tests/api_tests.cpp)
target_link_libraries(ling-api-tests PRIVATE ling_api)
add_test(NAME api COMMAND ling-api-tests)
