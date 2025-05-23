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
	rm -rf ${PREFIX}
	INCLUDE=${PREFIX}/include
	NPROC=2
	LIB=${PREFIX}/lib
	SRC=${PREFIX}/src
	mkdir -p ${SRC}
	cd ${PROJECT} && \
		mkdir -p build ${INCLUDE} &&
    cp common/*.h ${INCLUDE} && \
		cp common/*.hpp ${INCLUDE} && \
		# For tracking changes
		cp tools/server/server.cpp ${SRC} && \
		cd build
  if [[ -n "${XLLAMACPP_BUILD_CUDA}" ]]; then
  	echo "Building for CUDA"
	cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_LIBDIR=lib -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DGGML_CPU=OFF -DGGML_CUDA=ON -DGGML_CUDA_FORCE_MMQ=ON && \
    cmake --build . --config Release -j ${NPROC} && \
    cmake --install . --prefix ${PREFIX}
  elif [[ -n "${XLLAMACPP_BUILD_HIP}" ]]; then
	echo "Building for AMD GPU"
	cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_LIBDIR=lib -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DGGML_CPU=OFF -DAMDGPU_TARGETS="gfx1100;gfx1101;gfx1102;gfx1030;gfx1031;gfx1032" -DCMAKE_HIP_COMPILER="$(hipconfig -l)/clang" -DGGML_HIP_ROCWMMA_FATTN=ON -DGGML_HIP=ON && \
	cmake --build . --config Release -j ${NPROC} && \
    cmake --install . --prefix ${PREFIX}
  else
	if [[ "$(uname -s)" == "Darwin" ]]; then
		if [[ "$(uname -m)" == "x86_64" ]]; then
			echo "Building for Intel"
			cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_BUILD_RPATH="@loader_path" -DLLAMA_CURL=OFF -DGGML_METAL=OFF && \
			cmake --build . --config Release -j ${NPROC} && \
			cmake --install . --prefix ${PREFIX}
		else
			echo "Building for Apple Silicon"
			cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_BUILD_RPATH="@loader_path" -DGGML_METAL_USE_BF16=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_CURL=OFF && \
			cmake --build . --config Release -j ${NPROC} && \
			cmake --install . --prefix ${PREFIX}
		fi
	else
		echo "Building for non-MacOS CPU"
		cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_LIBDIR=lib -DLLAMA_CURL=OFF && \
		cmake --build . --config Release -j ${NPROC} && \
		cmake --install . --prefix ${PREFIX}
	fi
  fi
  [[ -e common/libcommon.a ]] && cp common/libcommon.a ${LIB}
  [[ -e common/Release/common.lib ]] && cp common/Release/common.lib ${LIB}
  [[ -e tools/mtmd/libmtmd_static.a ]] && cp tools/mtmd/libmtmd_static.a ${LIB}
  [[ -e tools/mtmd/Release/mtmd_static.lib ]] && cp tools/mtmd/Release/mtmd_static.lib ${LIB}
  cd ${CWD}
}

build_llamacpp
