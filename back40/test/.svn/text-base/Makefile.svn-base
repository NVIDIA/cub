#-------------------------------------------------------------------------------
#
# Copyright 2010-2012 Duane Merrill
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 
#  
#-------------------------------------------------------------------------------
 
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
# Tune arch
#-------------------------------------------------------------------------------

ifdef tunearch
    TUNE_ARCH = $(tunearch)
else 
	TUNE_ARCH = 200
endif

ifeq ($(TUNE_ARCH), 200) 
	TUNE_SM_TARGETS = $(GEN_SM20)
endif
ifeq ($(TUNE_ARCH), 130) 
	TUNE_SM_TARGETS = $(GEN_SM13)
endif
ifeq ($(TUNE_ARCH), 100) 
	TUNE_SM_TARGETS = $(GEN_SM10)
endif

#-------------------------------------------------------------------------------
# Tune size
#-------------------------------------------------------------------------------

ifdef tunesize
    TUNE_SIZE = $(tunesize)
else 
	TUNE_SIZE = 4
endif


#-------------------------------------------------------------------------------
# Libs
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------

INC = -I../..  

#-------------------------------------------------------------------------------
# Defines
#-------------------------------------------------------------------------------

DEFINES = 

#-------------------------------------------------------------------------------
# Compiler Flags
#-------------------------------------------------------------------------------

ifneq ($(force64), 1)
	# Compile with 32-bit device pointers by default
	ARCH_SUFFIX = i386
	ARCH = -m32
else
	ARCH_SUFFIX = x86_64
	ARCH = -m64
endif

NVCCFLAGS = -Xptxas -v -Xcudafe -\#

ifeq (WIN_NT, $(findstring WIN_NT, $(OSUPPER)))
	NVCCFLAGS += -Xcompiler /bigobj -Xcompiler /Zm500
endif

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

DEPS = 	./Makefile \
		../../cub/test/test_util.h \
		$(call rwildcard,../../cub/,*.cuh) \
		$(call rwildcard,../../back40/,*.cuh) 

		 

#-------------------------------------------------------------------------------
# test_gpu_radix_sort  
#-------------------------------------------------------------------------------

test_gpu_radix_sort : bin/test_gpu_radix_sort_$(SUFFIX)

bin/test_gpu_radix_sort_$(SUFFIX) : test_gpu_radix_sort.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_gpu_radix_sort_$(SUFFIX) test_gpu_radix_sort.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3  



#-------------------------------------------------------------------------------
# clean
#-------------------------------------------------------------------------------

clean :
	rm -f bin/*_$(SUFFIX)* 
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o




