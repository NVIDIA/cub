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
# Build script for project
#-------------------------------------------------------------------------------

NVCC = "$(shell which nvcc)"
NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])

#-------------------------------------------------------------------------------
# Gen targets
#-------------------------------------------------------------------------------

GEN_SM35 = -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GEN_SM30 = -gencode=arch=compute_30,code=\"sm_30,compute_30\"
GEN_SM20 = -gencode=arch=compute_20,code=\"sm_20,compute_20\"
GEN_SM13 = -gencode=arch=compute_13,code=\"sm_13,compute_13\" 
GEN_SM11 = -gencode=arch=compute_11,code=\"sm_11,compute_11\" 
GEN_SM10 = -gencode=arch=compute_10,code=\"sm_10,compute_10\" 
#SM_TARGETS = $(GEN_SM20) $(GEN_SM13) $(GEN_SM10)
SM_TARGETS = $(GEN_SM20)

#-------------------------------------------------------------------------------
# Libs
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------

INC = -I. -I../.. 

#-------------------------------------------------------------------------------
# Defines
#-------------------------------------------------------------------------------

DEFINES = 

#-------------------------------------------------------------------------------
# Compiler Flags
#-------------------------------------------------------------------------------

ifneq ($(force64), 1)
	# Compile with 32-bit device pointers by default
	ARCH = -m32
	ARCH_SUFFIX = i386
else
	ARCH = -m64
	ARCH_SUFFIX = x86_64
endif

NVCCFLAGS = -Xptxas -v -Xcudafe -\#

ifeq ($(abi), 1)
	ABI_SUFFIX = abi
else 
	# Disable the ABI by default
	NVCCFLAGS += -Xptxas -abi=no
	ABI_SUFFIX = noabi
endif

ifeq ($(open64), 1)
	# Use open64 compiler instead of nvvm
	NVCCFLAGS += -open64
	PTX_SUFFIX = open64
else 
	PTX_SUFFIX = nvvm
endif

ifeq ($(verbose), 1)
	NVCCFLAGS += -v
endif

ifeq ($(keep), 1)
	NVCCFLAGS += -keep
endif

SUFFIX = $(PTX_SUFFIX)_$(NVCC_VERSION)_$(ABI_SUFFIX)_$(ARCH_SUFFIX)

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

DEPS =	 ./Makefile \
		./test_util.h \
		$(call rwildcard,../../cub/,*.cuh)


#-------------------------------------------------------------------------------
# make clean
#-------------------------------------------------------------------------------

clean :
	rm -f bin/*$(ARCH_SUFFIX)* 
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o


#-------------------------------------------------------------------------------
# make test_allocator 
#-------------------------------------------------------------------------------

test_allocator: bin/test_allocator_$(SUFFIX)

bin/test_allocator_$(SUFFIX) : test_allocator.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_allocator_$(SUFFIX) test_allocator.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3
	
	
#-------------------------------------------------------------------------------
# make test_grid_barrier 
#-------------------------------------------------------------------------------

test_grid_barrier: bin/test_grid_barrier_$(SUFFIX)

bin/test_grid_barrier_$(SUFFIX) : test_grid_barrier.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_grid_barrier_$(SUFFIX) test_grid_barrier.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3  
	

#-------------------------------------------------------------------------------
# make test_block_reduce 
#-------------------------------------------------------------------------------

test_block_reduce: bin/test_block_reduce_$(SUFFIX)

bin/test_block_reduce_$(SUFFIX) : test_block_reduce.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_block_reduce_$(SUFFIX) test_block_reduce.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3  


#-------------------------------------------------------------------------------
# make test_warp_scan 
#-------------------------------------------------------------------------------

test_warp_scan: bin/test_warp_scan_$(SUFFIX)

bin/test_warp_scan_$(SUFFIX) : test_warp_scan.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_warp_scan_$(SUFFIX) test_warp_scan.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3  


#-------------------------------------------------------------------------------
# make test_block_scan 
#-------------------------------------------------------------------------------

test_block_scan: bin/test_block_scan_$(SUFFIX)

bin/test_block_scan_$(SUFFIX) : test_block_scan.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_block_scan_$(SUFFIX) test_block_scan.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3  


#-------------------------------------------------------------------------------
# make test_block_load_store 
#-------------------------------------------------------------------------------

test_block_load_store: bin/test_block_load_store_$(SUFFIX)

bin/test_block_load_store_$(SUFFIX) : test_block_load_store.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_block_load_store_$(SUFFIX) test_block_load_store.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3
	
	
#-------------------------------------------------------------------------------
# make test_block_radix_sort 
#-------------------------------------------------------------------------------

test_block_radix_sort: bin/test_block_radix_sort_$(SUFFIX)

bin/test_block_radix_sort_$(SUFFIX) : test_block_radix_sort.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_block_radix_sort_$(SUFFIX) test_block_radix_sort.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3



#-------------------------------------------------------------------------------
# make test_coo_spmv_double
#-------------------------------------------------------------------------------

test_coo_spmv_double: bin/test_coo_spmv_double_$(SUFFIX)

bin/test_coo_spmv_double_$(SUFFIX) : test_coo_spmv_double.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_coo_spmv_double_$(SUFFIX) test_coo_spmv_double.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3

#-------------------------------------------------------------------------------
# make test_coo_spmv
#-------------------------------------------------------------------------------

test_coo_spmv: bin/test_coo_spmv_$(SUFFIX)

bin/test_coo_spmv_$(SUFFIX) : test_coo_spmv.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_coo_spmv_$(SUFFIX) test_coo_spmv.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3

#-------------------------------------------------------------------------------
# make test_block_serialize
#-------------------------------------------------------------------------------

test_block_serialize: bin/test_block_serialize_$(SUFFIX)

bin/test_block_serialize_$(SUFFIX) : test_block_serialize.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_block_serialize_$(SUFFIX) test_block_serialize.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3

