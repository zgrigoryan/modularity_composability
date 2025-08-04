cmake_minimum_required(VERSION 3.15)
project(ModularityAndFusion CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
add_compile_options(-O3 -DNDEBUG)

# Enable native vector/FMA where possible
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-march=native)
elseif (MSVC)
  add_compile_options(/O2)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(core
  src/math_ops.cpp
  src/benchmark.cpp
)

add_executable(modfuse src/main.cpp)
target_link_libraries(modfuse PRIVATE core)
