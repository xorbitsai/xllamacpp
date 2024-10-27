# set path so `llama-cli` etc.. be in path
export PATH := $(PWD)/bin:$(PATH)


THIRDPARTY := $(PWD)/thirdparty
LLAMACPP := $(THIRDPARTY)/llama.cpp

ifeq ($(WITH_DYLIB),1)
	LIBLAMMA := $(LLAMACPP)/lib/libllama.dylib
else
	LIBLAMMA := $(LLAMACPP)/lib/libllama.a	
endif

MODEL := models/gemma-2-9b-it-IQ4_XS.gguf
# MODEL := models/mistral-7b-instruct-v0.1.Q4_K_M.gguf
# MODEL := models/Llama-3.2-1B-Instruct-Q6_K.gguf


MIN_OSX_VER := -mmacosx-version-min=13.6

.PHONY: cmake clean reset setup setup_inplace wheel bind header

all: build

$(LIBLAMMA):
	@scripts/setup.sh

build: $(LIBLAMMA)
	@python3 setup.py build_ext --inplace
	#@rm -rf build

wheel:
	@echo "WITH_DYLIB=$(WITH_DYLIB)"
	@python3 setup.py bdist_wheel
ifeq ($(WITH_DYLIB),1)
	delocate-wheel -v dist/*.whl 
endif

build/include:
	@scripts/header_utils.py --force-overwrite --output_dir build/include include

bind: build/include
	@rm -rf build/bind
	@make -f scripts/bind/bind.mk bind


.PHONY: test test_simple test_main test_retrieve test_model \
		test_cy test_pb test_pb_hl test_nb bump clean reset

test:
	@pytest

test_simple:
	@g++ -std=c++14 -o build/simple \
		-I $(LLAMACPP)/include -L $(LLAMACPP)/lib  \
		-framework Foundation -framework Accelerate \
		-framework Metal -framework MetalKit \
		$(LLAMACPP)/lib/libllama.a \
		$(LLAMACPP)/lib/libggml.a \
		$(LLAMACPP)/lib/libcommon.a \
		build/llama.cpp/examples/simple/simple.cpp
	@./build/simple -m $(MODEL) \
		-p "When did the French Revolution start?" -c 2048 -n 512

test_main:
	@g++ -std=c++14 -o build/main \
		-I $(LLAMACPP)/include -L $(LLAMACPP)/lib  \
		-framework Foundation -framework Accelerate \
		-framework Metal -framework MetalKit \
		$(LLAMACPP)/lib/libllama.a \
		$(LLAMACPP)/lib/libggml.a \
		$(LLAMACPP)/lib/libcommon.a \
		build/llama.cpp/examples/main/main.cpp
	@./build/main -m $(MODEL) --log-disable \
		-p "When did the French Revolution start?" -c 2048 -n 512

test_retrieve:
	@./bin/llama-retrieval --model models/all-MiniLM-L6-v2-Q5_K_S.gguf \
		--top-k 3 --context-file README.md \
		--context-file LICENSE \
		--chunk-size 100 \
		--chunk-separator .

$(MODEL):
	@mkdir -p models && cd models && \
		wget https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-IQ4_XS.gguf

test_model: $(MODEL)
	@./bin/llama-simple -c 2048 -n 512 -m $(MODEL) \
	-p "Number of planets in our solar system"

test_cy: 
	@cd tests && python3 simple.py

test_highlevel: 
	@cd tests && python3 highlevel.py

test_llava:
	@./bin/llama-llava-cli -m models/llava-llama-3-8b-v1_1-int4.gguf \
		--mmproj models/llava-llama-3-8b-v1_1-mmproj-f16.gguf \
		--image tests/media/flower.jpg -c 4096 -e \
		-p "<|start_header_id|>user<|end_header_id|>\n\n<image>\nDescribe this image<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

bump:
	@scripts/bump.sh

clean:
	@rm -rf build dist src/*.egg-info .pytest_cache

reset: clean
	@rm -rf bin thirdparty 


