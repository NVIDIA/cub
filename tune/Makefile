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
# Libs
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Includes
#-------------------------------------------------------------------------------

INC = -I. -I../cub -I../test

#-------------------------------------------------------------------------------
# Defines
#-------------------------------------------------------------------------------

DEFINES = 

#-------------------------------------------------------------------------------
# Tune arch
#-------------------------------------------------------------------------------

ifdef tunearch
	TUNE_ARCH = $(tunearch)
else 
    TUNE_ARCH = 200
endif

ifeq ($(TUNE_ARCH), 350) 
    TUNE_SM_TARGET = -gencode=arch=compute_35,code=\"sm_35,compute_35\"
endif
ifeq ($(TUNE_ARCH), 300) 
    TUNE_SM_TARGET = -gencode=arch=compute_30,code=\"sm_30,compute_30\"
endif
ifeq ($(TUNE_ARCH), 200) 
    TUNE_SM_TARGET = -gencode=arch=compute_20,code=\"sm_20,compute_20\"
endif
ifeq ($(TUNE_ARCH), 130) 
    TUNE_SM_TARGET = -gencode=arch=compute_13,code=\"sm_13,compute_13\" 
endif
ifeq ($(TUNE_ARCH), 110) 
    TUNE_SM_TARGET = -gencode=arch=compute_11,code=\"sm_11,compute_11\" 
endif
ifeq ($(TUNE_ARCH), 100) 
    TUNE_SM_TARGET = -gencode=arch=compute_10,code=\"sm_10,compute_10\" 
endif


#-------------------------------------------------------------------------------
# Compiler Flags
#-------------------------------------------------------------------------------

NVCCFLAGS = -Xptxas -v -Xcudafe -\#

ifneq ($(force64), 1)
	# Compile with 32-bit device pointers by default
	ARCH = -m32
	ARCH_SUFFIX = i386
else
	ARCH = -m64
	ARCH_SUFFIX = x86_64
endif

ifneq ($(abi), 0)
	# Enable the ABI by default
	ABI_SUFFIX = abi
else 
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

ifeq ($(cnp), 1)
	NVCCFLAGS += -rdc=true -lcudadevrt
endif


SUFFIX = SM$(TUNE_ARCH)_$(PTX_SUFFIX)_$(NVCC_VERSION)_$(ABI_SUFFIX)_$(ARCH_SUFFIX)

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

DEPS =	 ./Makefile \
		../test/test_util.h \
		$(call rwildcard,../cub/,*.cuh)


#-------------------------------------------------------------------------------
# make clean
#-------------------------------------------------------------------------------

clean :
	rm -f bin/*$(ARCH_SUFFIX)* 
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o



#-------------------------------------------------------------------------------
# make tune_device_reduce
#-------------------------------------------------------------------------------

tune_device_reduce: bin/tune_device_reduce_$(SUFFIX)

bin/tune_device_reduce_$(SUFFIX) : tune_device_reduce.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(TUNE_SM_TARGET) -o bin/tune_device_reduce_$(SUFFIX) tune_device_reduce.cu $(NVCCFLAGS) $(ARCH) $(INC) -O3 -DTUNE_ARCH=$(TUNE_ARCH)

