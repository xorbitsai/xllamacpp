#!/usr/bin/env python3

"""
create a stripped version of llama.h
"""

import os
import re

headers = """\
#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

"""

target = 'thirdparty/llama.cpp/include/llama-stripped.h'
os.system(f"g++ -E thirdparty/llama.cpp/include/llama.h | sed '/^#/d' | sed '/^$/N;/^\\n$/D' | clang-format --style=file:scripts/.clang-format > {target}")

with open(target) as f:
    lines = f.readlines()
    res = []
    start = False
    for i, line in enumerate(lines):
        if line.startswith('struct llama_model;'):
            start = True
        if start:
            if line == '}\n':
                continue
            if '__attribute__((deprecated' in line:
                continue
            res.append(line)

headers = [header + '\n' for header in headers.splitlines()]
res = headers + res
with open(target,'w') as f:
    f.writelines(res)
