// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Loads a plugin the way NCCL does (dlopen with RTLD_NOW, so every undefined symbol must resolve) and
// looks up each op table named on the command line.
#include <dlfcn.h>
#include <stdio.h>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s PLUGIN.so SYMBOL...\n", argv[0]); return 2; }
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }
    for (int i = 2; i < argc; ++i) {
        if (!dlsym(lib, argv[i])) { fprintf(stderr, "missing %s\n", argv[i]); return 1; }
        printf("loaded %s\n", argv[i]);
    }
    return 0;
}
