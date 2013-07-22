#/******************************************************************************
# * Copyright (c) 2011, Duane Merrill.  All rights reserved.
# * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
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
# make <target> [sm=<XXX,...>] [cnp=<0|1>] [force64=<0|1>] [abi=<0|1>] [open64=<0|1>] [verbose=<0|1>] [keep=<0|1>]
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

# Set CUB_CNP macro for programs (default: no)
ifeq ($(cnp), 1)
	DEFINES += -DCUB_CNP
	CNP_SUFFIX = cnp
    NVCCFLAGS += -rdc=true -lcudadevrt
else
	CNP_SUFFIX = nocnp
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
endif
ifeq (300, $(findstring 300, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_30,code=\"sm_30,compute_30\"
endif
ifeq (200, $(findstring 200, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_20,code=\"sm_20,compute_20\"
endif
ifeq (130, $(findstring 130, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_13,code=\"sm_13,compute_13\" 
endif
ifeq (110, $(findstring 110, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_11,code=\"sm_11,compute_11\" 
endif
ifeq (100, $(findstring 100, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_10,code=\"sm_10,compute_10\" 
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
else
    # For g++
    # Disable excess x86 floating point precision that can lead to results being labeled incorrectly
    NVCCFLAGS += -Xcompiler -ffloat-store
endif

# 32/64-bit (32-bit device pointers by default) 
ifneq ($(force64), 1)
	# 
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
SUFFIX = sm$(SM_ARCH)_$(PTX_SUFFIX)_$(NVCC_VERSION)_$(ABI_SUFFIX)_$(CNP_SUFFIX)_$(CPU_ARCH_SUFFIX)



#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------

INC += -I. -I.. 

#-------------------------------------------------------------------------------
# Libs
#-------------------------------------------------------------------------------

LIBS += -lcudart 


#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

DEPS =	./Makefile \
		./../test/test_util.h \
		$(call rwildcard,../cub/,*.cuh)


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
# make example_block_reduce
#-------------------------------------------------------------------------------

example_block_reduce: bin/example_block_reduce_$(SUFFIX)

bin/example_block_reduce_$(SUFFIX) : example_block_reduce.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/example_block_reduce_$(SUFFIX) example_block_reduce.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make example_block_scan_sum
#-------------------------------------------------------------------------------

example_block_scan_sum: bin/example_block_scan_sum_$(SUFFIX)

bin/example_block_scan_sum_$(SUFFIX) : example_block_scan_sum.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/example_block_scan_sum_$(SUFFIX) example_block_scan_sum.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make example_block_radix_sort
#-------------------------------------------------------------------------------

example_block_radix_sort: bin/example_block_radix_sort_$(SUFFIX)

bin/example_block_radix_sort_$(SUFFIX) : example_block_radix_sort.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/example_block_radix_sort_$(SUFFIX) example_block_radix_sort.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3
	
	
#-------------------------------------------------------------------------------
# make histo_study
#-------------------------------------------------------------------------------

histo_study: bin/histo_study_$(SUFFIX)

bin/histo_study_$(SUFFIX) : histo_study.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/histo_study_$(SUFFIX) histo_study.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3	


#-------------------------------------------------------------------------------
# make example_coo_spmv
#-------------------------------------------------------------------------------

example_coo_spmv: bin/example_coo_spmv_$(SUFFIX)

bin/example_coo_spmv_$(SUFFIX) : example_coo_spmv.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/example_coo_spmv_$(SUFFIX) example_coo_spmv.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


