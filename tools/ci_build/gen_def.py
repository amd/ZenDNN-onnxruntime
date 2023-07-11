#!/usr/bin/python3

# *******************************************************************************
# Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# ******************************************************************************/

import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", required=True, help="input symbol file")
    parser.add_argument("--output", required=True, help="output file")
    parser.add_argument("--output_source", required=True, help="output file")
    parser.add_argument("--version_file", required=True, help="VERSION_NUMBER file")
    parser.add_argument("--style", required=True, choices=["gcc", "vc", "xcode"])
    parser.add_argument("--config", required=True, nargs="+")
    return parser.parse_args()


args = parse_arguments()
print("Generating symbol file for %s" % str(args.config))
with open(args.version_file) as f:
    VERSION_STRING = f.read().strip()

print("VERSION:%s" % VERSION_STRING)

symbols = set()
for c in args.config:
    file_name = os.path.join(args.src_root, "core", "providers", c, "symbols.txt")
    with open(file_name) as file:
        for line in file:
            line = line.strip()  # noqa: PLW2901
            if line in symbols:
                print("dup symbol: %s", line)
                exit(-1)
            symbols.add(line)
symbols = sorted(symbols)

symbol_index = 1
with open(args.output, "w") as file:
    if args.style == "vc":
        file.write("LIBRARY\n")
        file.write("EXPORTS\n")
    elif args.style == "xcode":
        pass  # xcode compile don't has any header.
    else:
        file.write("VERS_%s {\n" % VERSION_STRING)
        file.write(" global:\n")

    for symbol in symbols:
        if args.style == "vc":
            file.write(" %s @%d\n" % (symbol, symbol_index))
        elif args.style == "xcode":
            file.write("_%s\n" % symbol)
        else:
            file.write("  %s;\n" % symbol)
        symbol_index += 1

    if args.style == "gcc":
        file.write(" local:\n")
        file.write("    *;\n")
        file.write("};   \n")

with open(args.output_source, "w") as file:
    file.write("#include <onnxruntime_c_api.h>\n")
    for c in args.config:
        # WinML adapter should not be exported in platforms other than Windows.
        # Exporting OrtGetWinMLAdapter is exported without issues using .def file when compiling for Windows
        # so it isn't necessary to include it in generated_source.c

        # external symbols are removed, xnnpack ep will be created via the standard ORT API.
        # https://github.com/microsoft/onnxruntime/pull/11798
        if c not in ("vitisai", "winml", "cuda", "migraphx", "qnn", "snpe", "xnnpack", "cann", "dnnl", "zendnn"):
            file.write(f"#include <core/providers/{c}/{c}_provider_factory.h>\n")
    file.write("void* GetFunctionEntryByName(const char* name){\n")
    for symbol in symbols:
        if symbol != "OrtGetWinMLAdapter":
            file.write(f'if(strcmp(name,"{symbol}") ==0) return (void*)&{symbol};\n')
    file.write("return NULL;\n")
    file.write("}\n")
