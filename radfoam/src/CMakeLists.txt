find_package(CUDAToolkit REQUIRED)

# Find NVML (NVIDIA Management Library) to call nvmlDeviceGetMemoryInfo
find_library(NVML_LIBRARY
    NAMES nvidia-ml nvml
    PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/local/nvidia/lib64
    REQUIRED
)
message(STATUS "Found NVML: ${NVML_LIBRARY}")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -D_USE_MATH_DEFINES)

list(APPEND
  RADFOAM_INCLUDES
  ${CMAKE_SOURCE_DIR}/external/submodules/termcolor/include
  ${CMAKE_SOURCE_DIR}/external/submodules/eigen
  ${CMAKE_SOURCE_DIR}/external/submodules/atomic_queue/include
  ${GLFW_INCLUDES}
)

list(APPEND RADFOAM_CXX_SOURCES
  viewer/viewer.cpp
  utils/batch_fetcher.cpp
)
list(APPEND RADFOAM_CUDA_SOURCES
  aabb_tree/aabb_tree.cu
  delaunay/delaunay.cu
  delaunay/sample_initial_tets.cu
  delaunay/growth_iteration.cu
  delaunay/delete_violations.cu
  delaunay/triangulation_ops.cu
  tracing/pipeline.cu
)

add_library(radfoam STATIC
  ${RADFOAM_CXX_SOURCES}
  ${RADFOAM_CUDA_SOURCES}
)
target_include_directories(radfoam PUBLIC ${RADFOAM_INCLUDES})

target_link_libraries(radfoam PUBLIC
  CUDA::cudart
  CUDA::cuda_driver
  ${GLFW_LIBRARY}
  gl3w
  imgui
  TBB::tbb
  ${NVML_LIBRARY}          # <— link in NVML
)

target_compile_options(radfoam PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -O3 -Wall -Wno-unknown-pragmas>
  -O3
)