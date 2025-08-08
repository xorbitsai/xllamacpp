# scripts/setup.sh [download_last_working] [release-tag]
#
# setup.sh 			: (default run) downloads, builds and install last working release of llama.cpp
# setup.sh 1 		: like default
# setup.sh 0    	: downloads, builds and install bleeding edge llama.cpp from repo
# setup.sh 1 <tag>	: downloads, builds and install <tag> release of llama.cpp

CWD=$(pwd)
THIRDPARTY=${CWD}/thirdparty

build_llamacpp() {
	echo "update from llama.cpp main repo"
	PROJECT=${THIRDPARTY}/llama.cpp
	PREFIX=${CWD}/src/llama.cpp
	NPROC=2
	cd ${PROJECT} && \
		mkdir -p build &&
		cd build
  # Base CMake arguments
  CMAKE_ARGS=(
    "-DBUILD_SHARED_LIBS=OFF"
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
    "-DCMAKE_INSTALL_LIBDIR=lib"
    "-DLLAMA_CURL=OFF"
  )
  
  # Add any additional CMake arguments from environment
  if [ -n "${CMAKE_ARGS}" ]; then
    CMAKE_ARGS+=(${CMAKE_ARGS})
  fi

  # Build targets
  TARGETS=("common" "llama" "ggml" "ggml-cpu" "mtmd")
  
  if [[ -n "${XLLAMACPP_BUILD_CUDA}" ]]; then
    echo "Building for CUDA"
    CMAKE_ARGS+=("-DGGML_NATIVE=OFF" "-DGGML_CUDA=ON" "-DGGML_CUDA_FORCE_MMQ=ON")
    TARGETS+=("ggml-cuda")
  elif [[ -n "${XLLAMACPP_BUILD_HIP}" ]]; then
    echo "Building for AMD GPU"
    CMAKE_ARGS+=(
      "-DGGML_NATIVE=OFF"
      "-DAMDGPU_TARGETS=gfx1100;gfx1101;gfx1102;gfx1030;gfx1031;gfx1032"
      "-DCMAKE_HIP_COMPILER=$(hipconfig -l)/clang"
      "-DGGML_HIP_ROCWMMA_FATTN=ON"
      "-DGGML_HIP=ON"
    )
    TARGETS+=("ggml-hip")
  elif [[ -n "${XLLAMACPP_BUILD_AARCH64}" ]]; then
    echo "Building for aarch64"
    CMAKE_ARGS+=("-DGGML_NATIVE=OFF" "-DGGML_CPU_ARM_ARCH=armv8-a")
  else
    if [[ "$(uname -s)" == "Darwin" ]]; then
      CMAKE_ARGS+=("-DCMAKE_BUILD_RPATH=@loader_path")
      if [[ "$(uname -m)" == "x86_64" ]]; then
        echo "Building for Intel"
        CMAKE_ARGS+=("-DGGML_METAL=OFF")
        TARGETS+=("ggml-blas")
      else
        echo "Building for Apple Silicon"
        CMAKE_ARGS+=("-DGGML_METAL_EMBED_LIBRARY=ON")
        TARGETS+=("ggml-blas" "ggml-metal")
      fi
    else
      echo "Building for non-MacOS CPU (optimize for native CPU)"
      # Check if BLAS is enabled in CMake args
      if [[ " ${CMAKE_ARGS[*]} " =~ " -DGGML_BLAS=ON " ]]; then
        echo "BLAS is enabled, adding ggml-blas to build targets"
        TARGETS+=("ggml-blas")
      fi
    fi
  fi

  # Run CMake and build
  cmake .. "${CMAKE_ARGS[@]}" && \
  # Build all targets in one command
  cmake --build . --config Release -j ${NPROC} --target ${TARGETS[*]}
  rm -rf ${PREFIX}
  python ${CWD}/scripts/copy_libs.py
  cd ${CWD}
}

build_llamacpp
