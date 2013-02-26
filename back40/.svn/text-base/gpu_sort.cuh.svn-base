/******************************************************************************
 * 
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
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
 * Radix sorting problem instance
 ******************************************************************************/

#pragma once

#include "../util/allocator.cuh"
#include "../util/basic_utils.cuh"
#include "../util/error_utils.cuh"
#include "../util/cta_progress.cuh"
#include "../util/ns_wrapper.cuh"
/*
#include "../radix_sort/sort_utils.cuh"
#include "../radix_sort/tuned_policy.cuh"

#include "../radix_sort/kernel/kernel_downsweep_pass.cuh"
#include "../radix_sort/kernel/kernel_hybrid_pass.cuh"
#include "../radix_sort/kernel/kernel_scan_pass.cuh"
#include "../radix_sort/kernel/kernel_single_tile.cuh"
#include "../radix_sort/kernel/kernel_upsweep_pass.cuh"
*/

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {



/******************************************************************************
 * Problem instance
 ******************************************************************************/

/**
 * Problem instance
 */
template <
	typename Allocator,
	typename KeyType,
	typename ValueType,
	typename SizeT>
struct ProblemInstance
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	/**
	 * Tuned pass policy whose type signature does not reflect the tuned
	 * SM architecture.
	 */
	template <
		ProblemSize 	PROBLEM_SIZE,
		int 			RADIX_BITS>
	struct OpaquePassPolicy
	{
		// The appropriate tuning arch-id from the arch-id targeted by the
		// active compiler pass.
		enum
		{
/*
			COMPILER_TUNE_ARCH 		= (__CUB_CUDA_ARCH__ >= 200) ?
										200 :
										(__CUB_CUDA_ARCH__ >= 130) ?
											130 :
											100
*/
			COMPILER_TUNE_ARCH = 200
		};

		// Tuned pass policy
		typedef TunedPassPolicy<
			COMPILER_TUNE_ARCH,
			ProblemInstance,
			PROBLEM_SIZE,
			RADIX_BITS> TunedPassPolicy;

		struct DispatchPolicy 	: TunedPassPolicy::DispatchPolicy {};
		struct UpsweepPolicy 	: TunedPassPolicy::UpsweepPolicy {};
		struct SpinePolicy 		: TunedPassPolicy::SpinePolicy {};
		struct DownsweepPolicy 	: TunedPassPolicy::DownsweepPolicy {};
		struct BinDescriptorPolicy 	: TunedPassPolicy::BinDescriptorPolicy {};
		struct TilePolicy 		: TunedPassPolicy::TilePolicy {};
	};


	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	Allocator			*allocator;

	KeyType				*d_keys[2];
	ValueType			*d_values[2];
	BinDescriptor		*d_bins[2];
	int 				selector;

	SizeT				num_elements;
	int 				low_bit;
	int					num_bits;
	cudaStream_t		stream;
	int			 		max_grid_size;
	bool				debug;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------


	/**
	 * Constructor
	 */
	ProblemInstance()
	{
		allocator = NULL;
		d_keys[0] = NULL;
		d_keys[1] = NULL;
		d_values[0] = NULL;
		d_values[1] = NULL;
		d_bins[0] = NULL;
		d_bins[1] = NULL;
	}


	/**
	 * Destructor
	 */
	virtual ~ProblemInstance()
	{
		if (allocator)
		{
			if (d_keys[0]) allocator.Deallocate(d_keys[1]);
			if (d_values[0]) allocator.Deallocate(d_values[1]);
			if (d_bins[0]) allocator.Deallocate(d_bins[0]);
			if (d_bins[1]) allocator.Deallocate(d_bins[1]);
		}
	}


	/**
	 * Initializer
	 */
	cudaError_t Init(
		util::CachedAllocator	*allocator,
		KeyType					*d_keys,
		ValueType				*d_values,
		SizeT					num_elements,
		int 					low_bit,
		int						num_bits,
		cudaStream_t			stream,
		int			 			max_grid_size,
		bool					debug)
	{
		cudaError_t error = cudaSuccess;

		do {

			this->selector = 0;
			this->d_keys[0] = d_keys;
			this->d_values[0] = d_values;

			// Allocate temporary keys and values arrays
			error = allocator->Allocate(this->d_keys[1], sizeof(KeyType) * num_elements);
			if (error) break;
			if (d_values != NULL)
			{
				error = allocator->Allocate(this->d_values[1], sizeof(ValueType) * num_elements);
				if (error) break;
			}

			// Allocate partition descriptor queues
			int max_partitions = (problem_instance.num_elements + partition_props.tile_elements - 1) / partition_props.tile_elements;
			error = allocator->Allocate(this->d_bins[0], sizeof(BinDescriptor) * num_elements);
			if (error) break;
			error = allocator->Allocate(this->d_bins[1], sizeof(BinDescriptor) * num_elements);
			if (error) break;

			// Initialize partition descriptor queues
			error = cudaMemSet(d_bins[0](), 0, sizeof(BinDescriptor) * max_partitions);
			if (util::B40CPerror(error, __FILE__, __LINE__)) break;
			error = cudaMemSet(d_bins[1](), 0, sizeof(BinDescriptor) * max_partitions);
			if (util::B40CPerror(error, __FILE__, __LINE__)) break;


			this->num_elements = num_elements;
			this->low_bit = low_bit;
			this->num_bits = num_bits;
			this->stream = stream;
			this->max_grid_size = max_grid_size;
			this->debug = debug;

		} while (0);
	}


	/**
	 * Dispatch global partition
	 */
	cudaError_t DispatchGlobal(
		unsigned int 					radix_bits,
		const UpsweepKernelProps 		&upsweep_props,
		const SpineKernelProps			&spine_props,
		const DownsweepKernelProps		&downsweep_props,
		bool							unform_grid_size,
		DynamicSmemConfig				dynamic_smem_config)
	{
		cudaError_t error = cudaSuccess;

		do {
			// Current bit
			int current_bit = low_bit + num_bits - radix_bits;

			// Compute sweep grid size
			int schedule_granularity = CUB_MAX(
				upsweep_props.tile_elements,
				downsweep_props.tile_elements);

			int sweep_grid_size = downsweep_props.OversubscribedGridSize(
				schedule_granularity,
				num_elements,
				max_grid_size);

			// Compute spine elements (rounded up to nearest tile size)
			SizeT spine_elements = CUB_ROUND_UP_NEAREST(
				(sweep_grid_size << radix_bits),			// Each CTA produces a partial for every radix digit
				(1 << spine_props.log_tile_elements));		// Number of partials per tile

			// Make sure our spine is big enough
			error = spine.Setup(sizeof(SizeT) * spine_elements);
			if (error) break;

			// Obtain a CTA work distribution
			util::CtaWorkDistribution<SizeT> work(
				num_elements,
				sweep_grid_size,
				schedule_granularity);

			// Grid size tuning
			int grid_size[3] = {sweep_grid_size, 1, sweep_grid_size};
			if (unform_grid_size)
			{
				// Make sure that all kernels launch the same number of CTAs
				grid_size[1] = grid_size[0];
			}

			// Smem allocation tuning
			int dynamic_smem[3] = {0, 0, 0};

			if (dynamic_smem_config == DYNAMIC_SMEM_UNIFORM)
			{
				// Pad with dynamic smem so all kernels get the same total smem allocation
				int max_static_smem = CUB_MAX(
					upsweep_props.kernel_attrs.sharedSizeBytes,
					CUB_MAX(
						spine_props.kernel_attrs.sharedSizeBytes,
						downsweep_props.kernel_attrs.sharedSizeBytes));

				dynamic_smem[0] = max_static_smem - upsweep_props.kernel_attrs.sharedSizeBytes;
				dynamic_smem[1] = max_static_smem - spine_props.kernel_attrs.sharedSizeBytes;
				dynamic_smem[2] = max_static_smem - downsweep_props.kernel_attrs.sharedSizeBytes;
			}
			else if (dynamic_smem_config == DYNAMIC_SMEM_LCM)
			{
				// Pad upsweep/downsweep with dynamic smem so kernel occupancy a multiple of the lowest occupancy
				int min_occupancy = CUB_MIN(upsweep_props.max_cta_occupancy, downsweep_props.max_cta_occupancy);
				dynamic_smem[0] = upsweep_props.SmemPadding(min_occupancy);
				dynamic_smem[2] = downsweep_props.SmemPadding(min_occupancy);
			}

			// Print debug info
			if (debug)
			{
				work.Print();
				printf(
					"Current bit(%d)\n"
					"Upsweep:   tile size(%d), occupancy(%d), grid_size(%d), threads(%d), dynamic smem(%d)\n"
					"Spine:     tile size(%d), occupancy(%d), grid_size(%d), threads(%d), dynamic smem(%d)\n"
					"Downsweep: tile size(%d), occupancy(%d), grid_size(%d), threads(%d), dynamic smem(%d)\n",
					current_bit,
					upsweep_props.tile_elements, upsweep_props.max_cta_occupancy, grid_size[0], upsweep_props.threads, dynamic_smem[0],
					(1 << spine_props.log_tile_elements), spine_props.max_cta_occupancy, grid_size[1], spine_props.threads, dynamic_smem[1],
					downsweep_props.tile_elements, downsweep_props.max_cta_occupancy, grid_size[2], downsweep_props.threads, dynamic_smem[2]);
				fflush(stdout);
			}


			//
			// Upsweep
			//

			// Set shared mem bank mode
			cudaSharedMemConfig old_sm_config;
			cudaDeviceGetSharedMemConfig(&old_sm_config);
			if (old_sm_config != upsweep_props.sm_bank_config)
				cudaDeviceSetSharedMemConfig(upsweep_props.sm_bank_config);



			// Upsweep reduction into spine
			upsweep_props.kernel_func<<<grid_size[0], upsweep_props.threads, dynamic_smem[0], stream>>>(
				(SizeT*) spine(),
				storage.d_keys[storage.selector],
				work,
				current_bit);

			if (debug) {
				error = cudaThreadSynchronize();
				if (error = util::B40CPerror(error, "Upsweep kernel failed ", __FILE__, __LINE__)) break;
			}

			//
			// Spine
			//

			// Set shared mem bank mode
			if (spine_props.sm_bank_config != upsweep_props.sm_bank_config)
				cudaDeviceSetSharedMemConfig(spine_props.sm_bank_config);

			// Spine scan
			spine_props.kernel_func<<<grid_size[1], spine_props.threads, dynamic_smem[1], stream>>>(
				(SizeT*) spine(),
				(SizeT*) spine(),
				spine_elements);

			if (debug) {
				error = cudaThreadSynchronize();
				if (error = util::B40CPerror(error, "Spine kernel failed ", __FILE__, __LINE__)) break;
			}

			//
			// Downsweep
			//

			// Set shared mem bank mode
			if (downsweep_props.sm_bank_config != spine_props.sm_bank_config)
				cudaDeviceSetSharedMemConfig(downsweep_props.sm_bank_config);

			// Downsweep scan from spine
			downsweep_props.kernel_func<<<grid_size[2], downsweep_props.threads, dynamic_smem[2], stream>>>(
				(BinDescriptor*) partitions[storage.selector ^ 1](),
				(SizeT *) spine(),
				storage.d_keys[storage.selector],
				storage.d_keys[storage.selector ^ 1],
				storage.d_values[storage.selector],
				storage.d_values[storage.selector ^ 1],
				work,
				current_bit);

			if (debug) {
				error = cudaThreadSynchronize();
				if (error = util::B40CPerror(error, "Downsweep kernel failed ", __FILE__, __LINE__)) break;
			}

			// Restore smem bank mode
			if (old_sm_config != downsweep_props.sm_bank_config)
				cudaDeviceSetSharedMemConfig(old_sm_config);

			// Update selector
			storage.selector ^= 1;

		} while(0);

		return error;
	}


	/**
	 * Dispatch partition sort
	 */
	cudaError_t DispatchBinDescriptor(
		const BinDescriptorKernelProps 	&partition_props,
		int 						initial_selector,
		int 						grid_size)
	{
		cudaError_t error = cudaSuccess;

		do {

			// Print debug info
			if (debug)
			{
				printf("BinDescriptor: tile size(%d), occupancy(%d), grid_size(%d), threads(%d)\n",
					partition_props.tile_elements,
					partition_props.max_cta_occupancy,
					grid_size,
					partition_props.threads);
				fflush(stdout);
			}

			// Set shared mem bank mode
			cudaSharedMemConfig old_sm_config;
			cudaDeviceGetSharedMemConfig(&old_sm_config);
			if (old_sm_config != partition_props.sm_bank_config)
				cudaDeviceSetSharedMemConfig(partition_props.sm_bank_config);

			// Tile sorting kernel
			partition_props.kernel_func<<<grid_size, partition_props.threads, 0, stream>>>(
				(BinDescriptor*) partitions[storage.selector](),
				(BinDescriptor*) partitions[storage.selector ^ 1](),
				storage.d_keys[storage.selector],
				storage.d_keys[storage.selector ^ 1],
				storage.d_keys[initial_selector],
				storage.d_values[storage.selector],
				storage.d_values[storage.selector ^ 1],
				storage.d_values[initial_selector],
				low_bit);

			if (debug) {
				error = cudaThreadSynchronize();
				if (error = util::B40CPerror(error, "Single kernel failed ", __FILE__, __LINE__)) break;
			}

			// Restore smem bank mode
			if (old_sm_config != partition_props.sm_bank_config)
				cudaDeviceSetSharedMemConfig(old_sm_config);

			// Update selector
			storage.selector ^= 1;

		} while(0);

		return error;
	}



	/**
	 * Dispatch single-CTA tile sort
	 */
	cudaError_t DispatchTile(const cta::SingleTileKernelProps &single_tile_props)
	{
		cudaError_t error = cudaSuccess;

		do {

			// Compute grid size
			int grid_size = 1;

			// Print debug info
			if (debug)
			{
				printf("Single tile: tile size(%d), occupancy(%d), grid_size(%d), threads(%d)\n",
					single_tile_props.tile_elements,
					single_tile_props.max_cta_occupancy,
					grid_size,
					single_tile_props.threads);
				fflush(stdout);
			}

			// Set shared mem bank mode
			cudaSharedMemConfig old_sm_config;
			cudaDeviceGetSharedMemConfig(&old_sm_config);
			if (old_sm_config != single_tile_props.sm_bank_config)
				cudaDeviceSetSharedMemConfig(single_tile_props.sm_bank_config);

			// Single-CTA sorting kernel
			single_tile_props.kernel_func<<<grid_size, single_tile_props.threads, 0, stream>>>(
				storage.d_keys[storage.selector],
				storage.d_values[storage.selector],
				low_bit,
				num_bits,
				num_elements);

			if (debug) {
				error = cudaThreadSynchronize();
				if (error = util::B40CPerror(error, "Single kernel failed ", __FILE__, __LINE__)) break;
			}

			// Restore smem bank mode
			if (old_sm_config != single_tile_props.sm_bank_config)
				cudaDeviceSetSharedMemConfig(old_sm_config);

		} while(0);

		return error;
	}





	/**
	 * Sort.
	 */
	template <int TUNE_ARCH, ProblemSize PROBLEM_SIZE>
	cudaError_t Sort()
	{
		cudaError_t error = cudaSuccess;
		do
		{
			enum
			{
				RADIX_BITS = PreferredDigitBits<TUNE_ARCH>::PREFERRED_BITS,
			};

			// Define tuned and opaque pass policies
			typedef radix_sort::TunedPassPolicy<TUNE_ARCH, ProblemInstance, PROBLEM_SIZE, RADIX_BITS> 	TunedPassPolicy;
			typedef OpaquePassPolicy<ProblemInstance, PROBLEM_SIZE, RADIX_BITS>							OpaquePassPolicy;

			int sm_version = cuda_props.device_sm_version;
			int sm_count = cuda_props.device_props.multiProcessorCount;

			// Upsweep kernel props
			kernel::UpsweepKernelProps<SizeT, KeyType> upsweep_props;
			error = upsweep_props.template Init<
				typename TunedPassPolicy::UpsweepPolicy,
				typename OpaquePassPolicy::UpsweepPolicy>(sm_version, sm_count);
			if (error) break;

			// Spine kernel props
			typename ProblemInstance::SpineKernelProps spine_props;
			error = spine_props.template Init<
				typename TunedPassPolicy::SpinePolicy,
				typename OpaquePassPolicy::SpinePolicy>(sm_version, sm_count);
			if (error) break;

			// Downsweep kernel props
			typename ProblemInstance::DownsweepKernelProps downsweep_props;
			error = downsweep_props.template Init<
				typename TunedPassPolicy::DownsweepPolicy,
				typename OpaquePassPolicy::DownsweepPolicy>(sm_version, sm_count);
			if (error) break;

			// Single-tile kernel props
			typename ProblemInstance::TileKernelProps single_tile_props;
			error = single_tile_props.template Init<
				typename TunedPassPolicy::TilePolicy,
				typename OpaquePassPolicy::TilePolicy>(sm_version, sm_count);
			if (error) break;

			// Hygrid kernel props
			typename ProblemInstance::BinDescriptorKernelProps partition_props;
			error = partition_props.template Init<
				typename TunedPassPolicy::BinDescriptorPolicy,
				typename OpaquePassPolicy::BinDescriptorPolicy>(sm_version, sm_count);
			if (error) break;

			//
			// Allocate
			//

			// Make sure our partition descriptor queues are big enough
			int max_partitions = (problem_instance.num_elements + partition_props.tile_elements - 1) / partition_props.tile_elements;
			size_t queue_bytes = sizeof(BinDescriptor) * max_partitions;

			error = partitions[0].Setup(queue_bytes);
			if (error) break;
			error = partitions[1].Setup(queue_bytes);
			if (error) break;



			//
			// First pass
			//

			// Print debug info
			if (problem_instance.debug)
			{
				printf("\nLow bit(%d), num bits(%d), radix_bits(%d), tuned arch(%d), SM arch(%d)\n",
					problem_instance.low_bit,
					problem_instance.num_bits,
					RADIX_BITS,
					TUNE_ARCH,
					cuda_props.device_sm_version);
				fflush(stdout);
			}

			// Dispatch first pass
			error = problem_instance.DispatchGlobal(
				RADIX_BITS,
				upsweep_props,
				spine_props,
				downsweep_props,
				TunedPassPolicy::DispatchPolicy::UNIFORM_GRID_SIZE,
				TunedPassPolicy::DispatchPolicy::DYNAMIC_SMEM_CONFIG);
			if (error) break;

			// Perform block iterations
			int grid_size = 32;
			{
				error = problem_instance.DispatchBinDescriptor(
					partition_props,
					initial_selector,
					grid_size);
				if (error) break;

				grid_size *= 32;
			}

			// Reset selector
			problem_instance.storage.selector = initial_selector;

		} while (0);

		return error;
	}


};


/**
 * Enact a sort.
 * @return cudaSuccess on success, error enumeration otherwise
 */
template <ProblemSize PROBLEM_SIZE, typename KeyType>
cudaError_t GpuSort(
	KeyType 		*d_keys,
	int 			num_elements,
	int				low_bit,
	int 			num_bits,
	cudaStream_t	stream 			= 0,
	int 			max_grid_size 	= 0,
	bool 			debug 			= false)
{
	typedef ProblemInstance<KeyType, util::NullType, int> ProblemInstance;

	if (num_elements <= 1)
	{
		// Nothing to do
		return cudaSuccess;
	}

	ProblemInstance<KeyType, util::NullType, int> problem_instance(
		d_keys,
		NULL,
		num_elements,
		low_bit,
		num_bits,
		stream,
		max_grid_size,
		debug);

//		if (cuda_props.kernel_ptx_version >= 200)
	{
		return problem_instance.Sort<200, PROBLEM_SIZE>();
	}
/*		else if (cuda_props.kernel_ptx_version >= 130)
	{
		return problem_instance.Sort<200, PROBLEM_SIZE>();
	}
	else
	{
		return problem_instance.Sort<200, PROBLEM_SIZE>();
	}
*/
}





} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
