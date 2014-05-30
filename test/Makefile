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
# make <target> [sm=<XXX,...>] [cdp=<0|1>] [force64=<0|1>] [abi=<0|1>] [open64=<0|1>] [verbose=<0|1>] [keep=<0|1>] [quicktest=<0|1>]
#
#-------------------------------------------------------------------------------
 



#-------------------------------------------------------------------------------
# Commandline Options
#-------------------------------------------------------------------------------

# Compute-capability to compile for, e.g., "sm=200,300,350" (SM20 by default).  
COMMA = ,
ifdef sm
	SM_ARCH = $(subst $(COMMA),-,$(sm))
else 
    SM_ARCH = 200
endif

ifeq (350, $(findstring 350, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_35,code=\"sm_35,compute_35\" 
    SM_DEF 		+= -DSM350
    TEST_ARCH 	= 350
endif
ifeq (300, $(findstring 300, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_30,code=\"sm_30,compute_30\"
    SM_DEF 		+= -DSM300
    TEST_ARCH 	= 300
endif
ifeq (210, $(findstring 210, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_20,code=\"sm_21,compute_20\"
    SM_DEF 		+= -DSM210
    TEST_ARCH 	= 210
endif
ifeq (200, $(findstring 200, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_20,code=\"sm_20,compute_20\"
    SM_DEF 		+= -DSM200
    TEST_ARCH 	= 200
endif
ifeq (130, $(findstring 130, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_13,code=\"sm_13,compute_13\" 
    SM_DEF 		+= -DSM130
    TEST_ARCH 	= 130
endif
ifeq (120, $(findstring 120, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_12,code=\"sm_12,compute_12\" 
    SM_DEF 		+= -DSM120
    TEST_ARCH 	= 120
endif
ifeq (110, $(findstring 110, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_11,code=\"sm_11,compute_11\" 
    SM_DEF 		+= -DSM110
    TEST_ARCH 	= 110
endif
ifeq (100, $(findstring 100, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_10,code=\"sm_10,compute_10\" 
    SM_DEF 		+= -DSM100
    TEST_ARCH 	= 100
endif


# CDP enable option (default: no)
ifeq ($(cdp), 1)
	DEFINES += -DCUB_CDP
	CDP_SUFFIX = cdp
    NVCCFLAGS += -rdc=true -lcudadevrt
else
	CDP_SUFFIX = nocdp
endif


# Testing mode option (quick/thorough)
ifeq ($(quicktest), 1)
	NVCCFLAGS += -DQUICK_TEST
	TEST_SUFFIX = quick
else 
	TEST_SUFFIX = thorough
endif


# CUDA memcheck (enabled by default) 
ifeq ($(memcheck), 0)
	MEMCHECK = 
else 
	MEMCHECK = cuda-memcheck
endif


# Device addressing mode option (32-bit device pointers by default) 
ifneq ($(force64), 1)
	CPU_ARCH = -m32
	CPU_ARCH_SUFFIX = i386
else
	CPU_ARCH = -m64
	CPU_ARCH_SUFFIX = x86_64
endif


# CUDA ABI option (enabled by default) 
ifneq ($(abi), 0)
	ABI_SUFFIX = abi
else 
	NVCCFLAGS += -Xptxas -abi=no
	ABI_SUFFIX = noabi
endif


# Middle-end compiler option (nvvm by default)
ifeq ($(open64), 1)
	NVCCFLAGS += -open64
	PTX_SUFFIX = open64
else 
	PTX_SUFFIX = nvvm
endif


# Verbose toolchain output from nvcc option
ifeq ($(verbose), 1)
	NVCCFLAGS += -v
endif


# Keep intermediate compilation artifacts option
ifeq ($(keep), 1)
	NVCCFLAGS += -keep
endif

# Debug mode
ifeq ($(debug), 1)
	NVCCFLAGS += -G
endif


#-------------------------------------------------------------------------------
# Compiler and compilation platform
#-------------------------------------------------------------------------------

NVCC = "$(shell which nvcc)"
ifdef nvccver
    NVCC_VERSION = $(nvccver)
else
    NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))
endif

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])


#-------------------------------------------------------------------------------
# Compiler Flags
#-------------------------------------------------------------------------------

# Default flags: verbose kernel properties (regs, smem, cmem, etc.); runtimes for compilation phases 
NVCCFLAGS += $(SM_DEF) -Xptxas -v -Xcudafe -\# 

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


# Includes
INC += -I. -I.. 


# Libs
#LIBS += -lcudart

# Define test arch
DEFINES += -DTEST_ARCH=$(TEST_ARCH)

# Suffix to append to each binary
SUFFIX = sm$(SM_ARCH)_$(PTX_SUFFIX)_$(NVCC_VERSION)_$(ABI_SUFFIX)_$(CDP_SUFFIX)_$(TEST_SUFFIX)_$(CPU_ARCH_SUFFIX)



#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

DEPS =				./test_util.h \
					./mersenne.h \
					$(call rwildcard,../cub/,*.cuh)
		

BLOCK_REDUCE = 		test_block_reduce_raking \
	 				test_block_reduce_warp_reductions		


BLOCK_SCAN = 		test_block_scan_raking \
	 				test_block_scan_raking_memoize \
	 				test_block_scan_warp_scans		


BLOCK_RADIX_SORT = 	test_block_radix_sort_keys \
	 				test_block_radix_sort_pairs	

		
ALL = 				link \
	 				test_iterator \
	 				test_allocator \
	 				test_warp_scan \
	 				test_warp_reduce \
	 				$(BLOCK_REDUCE) \
	 				$(BLOCK_SCAN) \
	 				$(BLOCK_RADIX_SORT) \
	 				test_block_load_store \
	 				test_block_histogram \
				 	test_device_reduce \
			 		test_device_histogram \
			 		test_device_scan \
			 		test_device_radix_sort \
					test_device_reduce_by_key\
		 			test_device_select_unique \
					test_device_select_flagged \
					test_device_select_if 
		
#	 	test_grid_barrier \		fails on sm110
#	 	test_device_seg_reduce
		


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
	for i in $(ALL); do $(MEMCHECK) ./bin/$${i}_$(SUFFIX) --device=$(device) || exit 1; done

run_block_reduce : 
	for i in $(BLOCK_REDUCE); do $(MEMCHECK) ./bin/$${i}_$(SUFFIX) --device=$(device) || exit 1; done

run_block_scan : 
	for i in $(BLOCK_SCAN); do $(MEMCHECK) ./bin/$${i}_$(SUFFIX) --device=$(device) || exit 1; done

run_block_radix_sort : 
	for i in $(BLOCK_RADIX_SORT); do $(MEMCHECK) ./bin/$${i}_$(SUFFIX) --device=$(device) || exit 1; done



#-------------------------------------------------------------------------------
# make link
#-------------------------------------------------------------------------------

link : bin/link_$(SUFFIX)

bin/link_$(SUFFIX) : link_a.cu link_b.cu link_main.cpp $(DEPS)
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(DEFINES) $(SM_TARGETS) link_a.cu -c -o bin/link_a.obj
	$(NVCC) $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(DEFINES) $(SM_TARGETS) link_b.cu -c -o bin/link_b.obj
	$(NVCC) $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(DEFINES) $(SM_TARGETS) link_main.cpp bin/link_a.obj bin/link_b.obj $(CUDART) -o bin/link_$(SUFFIX)


#-------------------------------------------------------------------------------
# make test_iterator 
#-------------------------------------------------------------------------------

test_iterator: bin/test_iterator_$(SUFFIX)

bin/test_iterator_$(SUFFIX) : test_iterator.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_iterator_$(SUFFIX) test_iterator.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make test_allocator 
#-------------------------------------------------------------------------------

test_allocator: bin/test_allocator_$(SUFFIX)

bin/test_allocator_$(SUFFIX) : test_allocator.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_allocator_$(SUFFIX) test_allocator.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3
	
	
#-------------------------------------------------------------------------------
# make test_grid_barrier 
#-------------------------------------------------------------------------------

test_grid_barrier: bin/test_grid_barrier_$(SUFFIX)

bin/test_grid_barrier_$(SUFFIX) : test_grid_barrier.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_grid_barrier_$(SUFFIX) test_grid_barrier.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3  
	

#-------------------------------------------------------------------------------
# make test_warp_scan 
#-------------------------------------------------------------------------------

test_warp_scan: bin/test_warp_scan_$(SUFFIX)

bin/test_warp_scan_$(SUFFIX) : test_warp_scan.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_warp_scan_$(SUFFIX) test_warp_scan.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3  


#-------------------------------------------------------------------------------
# make test_warp_reduce 
#-------------------------------------------------------------------------------

test_warp_reduce: bin/test_warp_reduce_$(SUFFIX)

bin/test_warp_reduce_$(SUFFIX) : test_warp_reduce.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_warp_reduce_$(SUFFIX) test_warp_reduce.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3  


#-------------------------------------------------------------------------------
# make test_block_reduce_raking
#-------------------------------------------------------------------------------

test_block_reduce_raking: bin/test_block_reduce_raking_$(SUFFIX)

bin/test_block_reduce_raking_$(SUFFIX) : test_block_reduce.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) -DTEST_RAKING $(SM_TARGETS) -o bin/test_block_reduce_raking_$(SUFFIX) test_block_reduce.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3  


#-------------------------------------------------------------------------------
# make test_block_reduce_warp_reductions 
#-------------------------------------------------------------------------------

test_block_reduce_warp_reductions: bin/test_block_reduce_warp_reductions_$(SUFFIX)

bin/test_block_reduce_warp_reductions_$(SUFFIX) : test_block_reduce.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) -DTEST_WARP_REDUCTIONS $(SM_TARGETS) -o bin/test_block_reduce_warp_reductions_$(SUFFIX) test_block_reduce.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3  


#-------------------------------------------------------------------------------
# make test_block_reduce 
#-------------------------------------------------------------------------------

test_block_reduce: $(BLOCK_REDUCE)


#-------------------------------------------------------------------------------
# make test_block_scan_raking
#-------------------------------------------------------------------------------

test_block_scan_raking: bin/test_block_scan_raking_$(SUFFIX)

bin/test_block_scan_raking_$(SUFFIX) : test_block_scan.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) -DTEST_RAKING $(SM_TARGETS) -o bin/test_block_scan_raking_$(SUFFIX) test_block_scan.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3  


#-------------------------------------------------------------------------------
# make test_block_scan_raking_memoize
#-------------------------------------------------------------------------------

test_block_scan_raking_memoize: bin/test_block_scan_raking_memoize_$(SUFFIX)

bin/test_block_scan_raking_memoize_$(SUFFIX) : test_block_scan.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) -DTEST_RAKING_MEMOIZE $(SM_TARGETS) -o bin/test_block_scan_raking_memoize_$(SUFFIX) test_block_scan.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3  


#-------------------------------------------------------------------------------
# make test_block_scan_warp_scans
#-------------------------------------------------------------------------------

test_block_scan_warp_scans: bin/test_block_scan_warp_scans_$(SUFFIX)

bin/test_block_scan_warp_scans_$(SUFFIX) : test_block_scan.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) -DTEST_WARP_SCANS $(SM_TARGETS) -o bin/test_block_scan_warp_scans_$(SUFFIX) test_block_scan.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3  


#-------------------------------------------------------------------------------
# make test_block_scan 
#-------------------------------------------------------------------------------

test_block_scan: $(BLOCK_SCAN)


#-------------------------------------------------------------------------------
# make test_block_load_store 
#-------------------------------------------------------------------------------

test_block_load_store: bin/test_block_load_store_$(SUFFIX)

bin/test_block_load_store_$(SUFFIX) : test_block_load_store.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_block_load_store_$(SUFFIX) test_block_load_store.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3
	
	
#-------------------------------------------------------------------------------
# make test_block_radix_sort_keys 
#-------------------------------------------------------------------------------

test_block_radix_sort_keys: bin/test_block_radix_sort_keys_$(SUFFIX)

bin/test_block_radix_sort_keys_$(SUFFIX) : test_block_radix_sort.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) -DTEST_KEYS_ONLY $(SM_TARGETS) -o bin/test_block_radix_sort_keys_$(SUFFIX) test_block_radix_sort.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3

#-------------------------------------------------------------------------------
# make test_block_radix_sort_pairs 
#-------------------------------------------------------------------------------

test_block_radix_sort_pairs: bin/test_block_radix_sort_pairs_$(SUFFIX)

bin/test_block_radix_sort_pairs_$(SUFFIX) : test_block_radix_sort.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_block_radix_sort_pairs_$(SUFFIX) test_block_radix_sort.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make test_block_radix_sort
#-------------------------------------------------------------------------------

test_block_radix_sort : $(BLOCK_RADIX_SORT)


#-------------------------------------------------------------------------------
# make test_block_histogram 
#-------------------------------------------------------------------------------

test_block_histogram: bin/test_block_histogram_$(SUFFIX)

bin/test_block_histogram_$(SUFFIX) : test_block_histogram.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_block_histogram_$(SUFFIX) test_block_histogram.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make test_device_reduce
#-------------------------------------------------------------------------------

test_device_reduce: bin/test_device_reduce_$(SUFFIX)

bin/test_device_reduce_$(SUFFIX) : test_device_reduce.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_reduce_$(SUFFIX) test_device_reduce.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make test_device_histogram
#-------------------------------------------------------------------------------

test_device_histogram: bin/test_device_histogram_$(SUFFIX)

bin/test_device_histogram_$(SUFFIX) : test_device_histogram.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_histogram_$(SUFFIX) test_device_histogram.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -lnppi -O3


#-------------------------------------------------------------------------------
# make test_device_scan
#-------------------------------------------------------------------------------

test_device_scan: bin/test_device_scan_$(SUFFIX)

bin/test_device_scan_$(SUFFIX) : test_device_scan.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_scan_$(SUFFIX) test_device_scan.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make test_device_radix_sort
#-------------------------------------------------------------------------------

test_device_radix_sort: bin/test_device_radix_sort_$(SUFFIX)

bin/test_device_radix_sort_$(SUFFIX) : test_device_radix_sort.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_radix_sort_$(SUFFIX) test_device_radix_sort.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make test_device_select_unique
#-------------------------------------------------------------------------------

test_device_select_unique: bin/test_device_select_unique_$(SUFFIX)

bin/test_device_select_unique_$(SUFFIX) : test_device_select_unique.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_select_unique_$(SUFFIX) test_device_select_unique.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


#-------------------------------------------------------------------------------
# make test_device_select_flagged
#-------------------------------------------------------------------------------

test_device_select_flagged: bin/test_device_select_flagged_$(SUFFIX)

bin/test_device_select_flagged_$(SUFFIX) : test_device_select_flagged.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_select_flagged_$(SUFFIX) test_device_select_flagged.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3

#-------------------------------------------------------------------------------
# make test_device_select_if
#-------------------------------------------------------------------------------

test_device_select_if: bin/test_device_select_if_$(SUFFIX)

bin/test_device_select_if_$(SUFFIX) : test_device_select_if.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_select_if_$(SUFFIX) test_device_select_if.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3

#-------------------------------------------------------------------------------
# make test_device_reduce_by_key
#-------------------------------------------------------------------------------

test_device_reduce_by_key: bin/test_device_reduce_by_key_$(SUFFIX)

bin/test_device_reduce_by_key_$(SUFFIX) : test_device_reduce_by_key.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_reduce_by_key_$(SUFFIX) test_device_reduce_by_key.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3





#-------------------------------------------------------------------------------
# make test_device_seg_reduce
#-------------------------------------------------------------------------------
#
#test_device_seg_reduce: bin/test_device_seg_reduce_$(SUFFIX)
#
#bin/test_device_seg_reduce_$(SUFFIX) : test_device_seg_reduce.cu $(DEPS)
#	mkdir -p bin
#	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_seg_reduce_$(SUFFIX) test_device_seg_reduce.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3


