<div align="center">
<img src="./assets/logo.png" width="400px" alt="xorbits" />

# xllamacpp - a Python wrapper of llama.cpp

[![PyPI Latest Release](https://img.shields.io/pypi/v/xllamacpp.svg?style=for-the-badge)](https://pypi.org/project/xllamacpp/)
[![License](https://img.shields.io/pypi/l/xllamacpp.svg?style=for-the-badge)](https://github.com/xorbitsai/inference/blob/main/LICENSE)
[![Discord](https://img.shields.io/badge/join_Discord-5462eb.svg?logo=discord&style=for-the-badge&logoColor=%23f5f5f5)](https://discord.gg/Xw9tszSkr5)
[![Twitter](https://img.shields.io/twitter/follow/xorbitsio?logo=x&style=for-the-badge)](https://twitter.com/xorbitsio)

</div>
<br />

This project forks from [cyllama](https://github.com/shakfu/cyllama) and provides a Python wrapper for @ggerganov's [llama.cpp](https://github.com/ggerganov/llama.cpp) which is likely the most active open-source compiled LLM inference engine.

## Compare to llama-cpp-python 

The following table provide an overview of the current implementations / features:

| implementations / features |      xllamacpp      |         llama-cpp-python         |
|:---------------------------|:-------------------:|:--------------------------------:|
| Wrapper-type               |       cython        |              ctypes              |
| API                        | Server & Params API |            Llama API             |
| Server implementation      |         C++         | Python through wrapped LLama API |
| Continuous batching        |         yes         |                no                |
| Thread safe                |         yes         |                no                |
| Release package            |      prebuilt       |    build during installation     |

It goes without saying that any help / collaboration / contributions to accelerate the above would be welcome!

## Wrapping Guidelines

As the intent is to provide a very thin wrapping layer and play to the strengths of the original c++ library as well as python, the approach to wrapping intentionally adopts the following guidelines:

- In general, key structs are implemented as cython extension classses with related functions implemented as methods of said classes.

- Be as consistent as possible with llama.cpp's naming of its api elements, except when it makes sense to shorten functions names which are used as methods.

- Minimize non-wrapper python code.

## Prerequisites for Prebuilt Wheels

Before pip installing `xllamacpp`, please ensure your system meets the following requirements based on your build type:

- **CPU (aarch64)**:
  - Requires ARMv8-A or later architecture
  - For best performance, build from source if your CPU supports advanced instruction sets

- **CUDA (Linux)**:
  - Requires glibc 2.35 or later
  - Compatible NVIDIA GPU with appropriate drivers (CUDA 12.4 or 12.8)

- **ROCm (Linux)**:
  - Requires glibc 2.35 or later
  - Requires gcc 10 or later (ROCm libraries have this dependency)
  - Compatible AMD GPU with ROCm support (ROCm 6.3.4 or 6.4.1)

## Install

**Note on Performance and Compatibility**

For maximum performance, you can build `xllamacpp` from source to optimize for your specific native CPU architecture. The pre-built wheels are designed for broad compatibility.

Specifically, the `aarch64` wheels are built for the `armv8-a` architecture. This ensures they run on a wide range of ARM64 devices, but it means that more advanced CPU instruction sets (like SVE) are not enabled. If your CPU supports these advanced features, building from source will provide better performance.

- From pypi for `CPU` or `Mac`:

```sh
pip install -U xllamacpp
```

- From github pypi for `CUDA` (use `--force-reinstall` to replace the installed CPU version):

  - CUDA 12.4
    ```sh
    pip install xllamacpp --force-reinstall --index-url https://xorbitsai.github.io/xllamacpp/whl/cu124 --extra-index-url https://pypi.org/simple
    ```

  - CUDA 12.8
    ```sh
    pip install xllamacpp --force-reinstall --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128 --extra-index-url https://pypi.org/simple
    ```

- From github pypi for `HIP` AMD GPU (use `--force-reinstall` to replace the installed CPU version):

  - ROCm 6.3.4
    ```sh
    pip install xllamacpp --force-reinstall --index-url https://xorbitsai.github.io/xllamacpp/whl/rocm-6.3.4 --extra-index-url https://pypi.org/simple
    ```

  - ROCm 6.4.1
    ```sh
    pip install xllamacpp --force-reinstall --index-url https://xorbitsai.github.io/xllamacpp/whl/rocm-6.4.1 --extra-index-url https://pypi.org/simple
    ```

## Build from Source

### (Optional) Preparation

- CUDA

  This provides GPU acceleration using an NVIDIA GPU. Make sure to have the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) installed.
  
  #### Download directly from NVIDIA
  You may find the official downloads here: [NVIDIA developer site](https://developer.nvidia.com/cuda-downloads).
  
  
  #### Compile and run inside a Fedora Toolbox Container
  We also have a [guide](./backend/CUDA-FEDORA.md) for setting up CUDA toolkit in a Fedora [toolbox container](https://containertoolbx.org/).
  
  **Recommended for:**
  - ***Necessary*** for users of [Atomic Desktops for Fedora](https://fedoraproject.org/atomic-desktops/); such as: [Silverblue](https://fedoraproject.org/atomic-desktops/silverblue/) and [Kinoite](https://fedoraproject.org/atomic-desktops/kinoite/).
    - (there are no supported CUDA packages for these systems)
  - ***Necessary*** for users that have a host that is not a: [Supported Nvidia CUDA Release Platform](https://developer.nvidia.com/cuda-downloads).
    - (for example, you may have [Fedora 42 Beta](https://fedoramagazine.org/announcing-fedora-linux-42-beta/) as your your host operating system)
  - ***Convenient*** For those running [Fedora Workstation](https://fedoraproject.org/workstation/) or [Fedora KDE Plasma Desktop](https://fedoraproject.org/spins/kde), and want to keep their host system clean.
  - *Optionally* toolbox packages are available: [Arch Linux](https://archlinux.org/), [Red Hat Enterprise Linux >= 8.5](https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux), or [Ubuntu](https://ubuntu.com/download)

- HIP

  This provides GPU acceleration on HIP-supported AMD GPUs.
  Make sure to have ROCm installed.
  You can download it from your Linux distro's package manager or from here: [ROCm Quick Start (Linux)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html#rocm-install-quick).

  
  Or you can try to build inside the [ROCm docker container](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html).

### Build `xllamacpp`

1. A recent version of `python3` (testing on python 3.12)

2. Git clone the latest version of `xllamacpp`:

 ```sh
 git clone git@github.com:xorbitsai/xllamacpp.git
 cd xllamacpp
 git submodule init
 git submodule update
 ```

3. Install dependencies of `cython`, `setuptools`, and `pytest` for testing:

 ```sh
 pip install -r requirements.txt
 ```

4. Type `CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" make` in the terminal to build xllamacpp with GGML_BLAS=ON.

## Testing

The `tests` directory in this repo provides extensive examples of using xllamacpp.

However, as a first step, you should download a smallish llm in the `.gguf` model from [huggingface](https://huggingface.co/models?search=gguf). A good model to start and which is assumed by tests is [Llama-3.2-1B-Instruct-Q8_0.gguf](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf). `xllamacpp` expects models to be stored in a `models` folder in the cloned `xllamacpp` directory. So to create the `models` directory if doesn't exist and download this model, you can just type:

```sh
make download
```

This basically just does:

```sh
cd xllamacpp
mkdir models && cd models
wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf 
```

Now you can test it using `llama-cli` or `llama-simple`:

```sh
bin/llama-cli -c 512 -n 32 -m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
 -p "Is mathematics discovered or invented?"
```

You can also run the test suite with `pytest` by typing `pytest` or:

```sh
make test
```
