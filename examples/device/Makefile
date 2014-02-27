#/******************************************************************************
# * Copyright (c) 2011, Duane Merrill.  All rights reserved.
# * Copyright (c) 2011-2014, NVIDIA CORPORATION.  All rights reserved.
# * 
# * Redistribution and use in source and binary forms, with or without
# * modification, are permitted provided that the following conditions are met:
# *	 * Redistributions of source code must retain the above copyright
# *	   notice, this list of conditions and the following disclaimer.
# *	 * Redistributions in binary form must reproduce the above copyright
# *	   notice, this list of conditions and the following disclaimer in the
# *	   documentation and/or other materials provided with the distribution.
# *	 * Neither the name of the NVIDIA CORPORATION nor the
# *	   names of its contributors may be used to endorse or promote products
# *	   derived from this software without specific prior written permission.
# * 
# * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *
#******************************************************************************/

#-------------------------------------------------------------------------------
#
# Makefile usage
#
# make <target> [sm=<XXX,...>] [cdp=<0|1>] [force64=<0|1>] [abi=<0|1>] [open64=<0|1>] [verbose=<0|1>] [keep=<0|1>]
#
#-------------------------------------------------------------------------------
 
#-------------------------------------------------------------------------------
# Compiler and compilation platform
#-------------------------------------------------------------------------------

NVCC = "$(shell which nvcc)"
NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])


#-------------------------------------------------------------------------------
# Defines
#-------------------------------------------------------------------------------

# Set CUB_CDP macro for programs (default: no)
ifeq ($(cdp), 1)
	DEFINES += -DCUB_CDP
	CDP_SUFFIX = cdp
    NVCCFLAGS += -rdc=true -lcudadevrt
else
	CDP_SUFFIX = nocdp
endif

#-------------------------------------------------------------------------------
# SM Arch
#-------------------------------------------------------------------------------

# Architecture(s) to compile for, e.g., "sm=200,300,350" (SM20 by default).  
COMMA = ,
ifdef sm
	SM_ARCH = $(subst $(COMMA),-,$(sm))
else 
    SM_ARCH = 200
endif

ifeq (350, $(findstring 350, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_35,code=\"sm_35,compute_35\" 
    SM_DEF += -DSM350
endif
ifeq (300, $(findstring 300, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_30,code=\"sm_30,compute_30\"
    SM_DEF += -DSM300
endif
ifeq (200, $(findstring 200, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_20,code=\"sm_20,compute_20\"
    SM_DEF += -DSM200
endif
ifeq (130, $(findstring 130, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_13,code=\"sm_13,compute_13\" 
    SM_DEF += -DSM130
endif
ifeq (120, $(findstring 120, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_12,code=\"sm_12,compute_12\" 
    SM_DEF += -DSM120
endif
ifeq (110, $(findstring 110, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_11,code=\"sm_11,compute_11\" 
    SM_DEF += -DSM110
endif
ifeq (100, $(findstring 100, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_10,code=\"sm_10,compute_10\" 
    SM_DEF += -DSM100
endif


#-------------------------------------------------------------------------------
# Compiler Flags
#-------------------------------------------------------------------------------

# Default flags: verbose kernel properties (regs, smem, cmem, etc.); runtimes for compilation phases 
NVCCFLAGS += -Xptxas -v -Xcudafe -\#

ifeq (WIN_NT, $(findstring WIN_NT, $(OSUPPER)))
    # For MSVC
    # Disable excess x86 floating point precision that can lead to results being labeled incorrectly
    NVCCFLAGS += -Xcompiler /fp:strict
    # Help the compiler/linker work with huge numbers of kernels on Windows
	NVCCFLAGS += -Xcompiler /bigobj -Xcompiler /Zm500
	CC = cl
ifneq ($(force64), 1)
	CUDART_CYG = "$(shell dirname $(NVCC))/../lib/Win32/cudart.lib"
else
	CUDART_CYG = "$(shell dirname $(NVCC))/../lib/x64/cudart.lib"
endif
	CUDART = "$(shell cygpath -w $(CUDART_CYG))"
else
    # For g++
    # Disable excess x86 floating point precision that can lead to results being labeled incorrectly
    NVCCFLAGS += -Xcompiler -ffloat-store
    CC = g++
ifneq ($(force64), 1)
    CUDART = "$(shell dirname $(NVCC))/../lib/libcudart_static.a"
else
    CUDART = "$(shell dirname $(NVCC))/../lib64/libcudart_static.a"
endif
endif

# 32/64-bit (32-bit device pointers by default) 
ifneq ($(force64), 1)
	CPU_ARCH = -m32
	CPU_ARCH_SUFFIX = i386
else
	CPU_ARCH = -m64
	CPU_ARCH_SUFFIX = x86_64
endif

# CUDA ABI enable/disable (enabled by default) 
ifneq ($(abi), 0)
	ABI_SUFFIX = abi
else 
	NVCCFLAGS += -Xptxas -abi=no
	ABI_SUFFIX = noabi
endif

# NVVM/Open64 middle-end compiler (nvvm by default)
ifeq ($(open64), 1)
	NVCCFLAGS += -open64
	PTX_SUFFIX = open64
else 
	PTX_SUFFIX = nvvm
endif

# Verbose toolchain output from nvcc
ifeq ($(verbose), 1)
	NVCCFLAGS += -v
endif

# Keep intermediate compilation artifacts
ifeq ($(keep), 1)
	NVCCFLAGS += -keep
endif

# Suffix to append to each binary
SUFFIX = sm$(SM_ARCH)_$(PTX_SUFFIX)_$(NVCC_VERSION)_$(ABI_SUFFIX)_$(CDP_SUFFIX)_$(CPU_ARCH_SUFFIX)



#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------

INC += -I. -I../.. -I../../test


#-------------------------------------------------------------------------------
# Libs
#-------------------------------------------------------------------------------

LIBS += -lcudart


#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

DEPS =	./Makefile \
		../../test/test_util.h \
		$(call rwildcard,../../cub/,*.cuh)
		
ALL = 	example_device_histogram \
		example_device_partition_flagged \
		example_device_partition_if \
	 	example_device_radix_sort \
		example_device_reduce \
	 	example_device_scan \
	 	example_device_select_unique \
		example_device_select_flagged \
		example_device_select_if
		


#-------------------------------------------------------------------------------
# make default
#-------------------------------------------------------------------------------

default:


#-------------------------------------------------------------------------------
# make clean
#-------------------------------------------------------------------------------

clean :
	rm -f bin/*$(CPU_ARCH_SUFFIX)* 
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o


#-------------------------------------------------------------------------------
# make all
#-------------------------------------------------------------------------------

all : $(ALL)

#-------------------------------------------------------------------------------
# make run
#-------------------------------------------------------------------------------

run : 
	for i in $(ALL); do ./bin/$${i}_$(SUFFIX) --device=$(device) || exit 1; done


#-------------------------------------------------------------------------------
# make example_device_reduce
#-------------------------------------------------------------------------------

example_device_reduce: bin/example_device_reduce_$(SUFFIX)

bin/example_device_reduce_$(SUFFIX) : example_device_reduce.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/example_device_reduce_$(SUFFIX) example_device_reduce.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make example_device_histogram
#-------------------------------------------------------------------------------

example_device_histogram: bin/example_device_histogram_$(SUFFIX)

bin/example_device_histogram_$(SUFFIX) : example_device_histogram.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/example_device_histogram_$(SUFFIX) example_device_histogram.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make example_device_partition_flagged
#-------------------------------------------------------------------------------

example_device_partition_flagged: bin/example_device_partition_flagged_$(SUFFIX)

bin/example_device_partition_flagged_$(SUFFIX) : example_device_partition_flagged.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/example_device_partition_flagged_$(SUFFIX) example_device_partition_flagged.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3

#-------------------------------------------------------------------------------
# make example_device_partition_if
#-------------------------------------------------------------------------------

example_device_partition_if: bin/example_device_partition_if_$(SUFFIX)

bin/example_device_partition_if_$(SUFFIX) : example_device_partition_if.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/example_device_partition_if_$(SUFFIX) example_device_partition_if.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3

#-------------------------------------------------------------------------------
# make example_device_scan
#-------------------------------------------------------------------------------

example_device_scan: bin/example_device_scan_$(SUFFIX)

bin/example_device_scan_$(SUFFIX) : example_device_scan.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/example_device_scan_$(SUFFIX) example_device_scan.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make example_device_radix_sort
#-------------------------------------------------------------------------------

example_device_radix_sort: bin/example_device_radix_sort_$(SUFFIX)

bin/example_device_radix_sort_$(SUFFIX) : example_device_radix_sort.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/example_device_radix_sort_$(SUFFIX) example_device_radix_sort.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make example_device_select_unique
#-------------------------------------------------------------------------------

example_device_select_unique: bin/example_device_select_unique_$(SUFFIX)

bin/example_device_select_unique_$(SUFFIX) : example_device_select_unique.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/example_device_select_unique_$(SUFFIX) example_device_select_unique.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make example_device_select_flagged
#-------------------------------------------------------------------------------

example_device_select_flagged: bin/example_device_select_flagged_$(SUFFIX)

bin/example_device_select_flagged_$(SUFFIX) : example_device_select_flagged.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/example_device_select_flagged_$(SUFFIX) example_device_select_flagged.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3

#-------------------------------------------------------------------------------
# make example_device_select_if
#-------------------------------------------------------------------------------

example_device_select_if: bin/example_device_select_if_$(SUFFIX)

bin/example_device_select_if_$(SUFFIX) : example_device_select_if.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/example_device_select_if_$(SUFFIX) example_device_select_if.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


