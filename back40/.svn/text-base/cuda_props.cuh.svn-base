/******************************************************************************
 * 
 * Copyright (c) 2011-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 ******************************************************************************/

/******************************************************************************
 * CUDA properties of the bundled "fatbin" assembly and attached devices
 ******************************************************************************/

#pragma once

#include <cub/perror.cuh>
#include <cub/device_props.cuh>

namespace cub {


/**
 * Invalid CUDA gpu device ordinal
 */
#define CUB_GPU_ORDINAL				(-1)


/**
 * Empty Kernel
 */
template <typename T>
__global__ void EmptyKernel(void) { }


/**
 * Encapsulation of device properties for a specific device
 */
class CudaProps
{
public:

	// Version information
	int 	sm_version;				// SM version of target device (SM version X.YZ in XYZ integer form)
	int 	ptx_version;			// Bundled PTX version for target device (PTX version X.YZ in XYZ integer form)

	// Target device properties
	int 	sm_count;				// Number of SMs
	int 	warp_threads;			// Number of threads per warp
	int 	smem_bank_bytes;		// Number of bytes per SM bank
	int 	smem_banks;				// Number of smem banks
	int 	smem_bytes;				// Smem bytes per SM
	int 	smem_alloc_unit;		// Smem segment size
	bool 	regs_by_block;			// Whether registers are allocated by CTA (or by warp)
	int 	reg_alloc_unit;			// Granularity of register allocation within the SM
	int 	warp_alloc_unit;		// Granularity of warp allocation within the SM
	int 	max_sm_threads;			// Maximum number of threads per SM
	int 	max_sm_ctas;			// Maximum number of CTAs per SM
	int 	max_cta_threads;		// Maximum number of threads per CTA
	int 	max_sm_registers;		// Maximum number of registers per SM
	int 	max_sm_warps;			// Maximum number of warps per SM


	/**
	 * Callback for initializing device properties
	 */
	template <typename StaticDeviceProps>
	void Callback()
	{
		warp_threads 		= StaticDeviceProps::WARP_THREADS;
		smem_bank_bytes		= StaticDeviceProps::SMEM_BANK_BYTES;
		smem_banks			= StaticDeviceProps::SMEM_BANKS;
		smem_bytes			= StaticDeviceProps::SMEM_BYTES;
		smem_alloc_unit		= StaticDeviceProps::SMEM_ALLOC_UNIT;
		regs_by_block		= StaticDeviceProps::REGS_BY_BLOCK;
		reg_alloc_unit		= StaticDeviceProps::REG_ALLOC_UNIT;
		warp_alloc_unit		= StaticDeviceProps::WARP_ALLOC_UNIT;
		max_sm_threads		= StaticDeviceProps::MAX_SM_THREADS;
		max_sm_ctas			= StaticDeviceProps::MAX_SM_CTAS;
		max_cta_threads		= StaticDeviceProps::MAX_CTA_THREADS;
		max_sm_registers	= StaticDeviceProps::MAX_SM_REGISTERS;
		max_sm_warps 		= max_sm_threads / warp_threads;
	}

public:

	/**
	 * Initializer.  Properties are retrieved for the specified GPU ordinal.
	 */
	static cudaError_t Init(CudaProps *cuda_props, int gpu_ordinal)
	{
		cudaError_t error = cudaSuccess;

		do {
			// Obtain SM version and count
			cudaDeviceProp device_props;
			if (error = Perror(cudaGetDeviceProperties(&device_props, gpu_ordinal),
				"cudaGetDeviceProperties failed", __FILE__, __LINE__)) break;
			cuda_props->sm_version = device_props.major * 100 + device_props.minor * 10;
			cuda_props->sm_count = device_props.multiProcessorCount;

			// Obtain PTX version of the bundled kernel assemblies compiled for
			// the current device
			cudaFuncAttributes flush_kernel_attrs;
			if (error = Perror(cudaFuncGetAttributes(&flush_kernel_attrs, EmptyKernel<void>),
				"cudaFuncGetAttributes failed", __FILE__, __LINE__)) break;
			cuda_props->ptx_version = flush_kernel_attrs.ptxVersion * 10;

			// Initialize our device properties via callback from static device properties
			StaticDeviceProps<100>::Callback(*cuda_props, cuda_props->sm_version);

		} while (0);

		return error;
	}


	/**
	 * Constructor.  Properties are retrieved for the current GPU ordinal.
	 */
	CudaProps()
	{
		do {
			int gpu_ordinal;
			if (cudaGetDevice(&gpu_ordinal)) break;
			if (Init(this, gpu_ordinal)) break;
		} while (0);
	}

	/**
	 * Constructor.  Properties are retrieved for the specified GPU ordinal.
	 */
	CudaProps(int gpu_ordinal)
	{
		Init(this, gpu_ordinal);
	}
};





} // namespace cub

