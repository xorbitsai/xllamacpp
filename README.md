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

| implementations / features             | xllamacpp     | llama-cpp-python |
| :--------------------------- | :-----------: | :--------------: |     
| Wrapper-type                 | cython        | ctypes           |
| API                           | Server & Params API  | Llama API |
| Server implementation   | C++           | Python through wrapped LLama API |
| Continuous batching    | yes           | no |
| Thread safe     | yes           | no |

It goes without saying that any help / collaboration / contributions to accelerate the above would be welcome!

## Wrapping Guidelines

As the intent is to provide a very thin wrapping layer and play to the strengths of the original c++ library as well as python, the approach to wrapping intentionally adopts the following guidelines:

- In general, key structs are implemented as cython extension classses with related functions implemented as methods of said classes.

- Be as consistent as possible with llama.cpp's naming of its api elements, except when it makes sense to shorten functions names which are used as methods.

- Minimize non-wrapper python code.

## Install

- From pypi for `CPU` or `Mac`:

```sh
pip install -U xllamacpp
```

- From github pypi for `CUDA` (use `--force-reinstall` to replace the installed CPU version):

```sh
pip install xllamacpp --force-reinstall --find-links https://xorbitsai.github.io/xllamacpp/whl/cu124
```

- From github pypi for `HIP` AMD GPU (use `--force-reinstall` to replace the installed CPU version):

```sh
pip install xllamacpp --force-reinstall --find-links https://xorbitsai.github.io/xllamacpp/whl/rocm-6.2.4
```

## Setup

To build `xllamacpp`:

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

4. Type `make` in the terminal.

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
