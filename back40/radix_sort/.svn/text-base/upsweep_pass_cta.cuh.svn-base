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
 * CTA-wide "upsweep" abstraction for computing radix digit histograms
 ******************************************************************************/

#pragma once

#include "../../util/basic_utils.cuh"
#include "../../util/device_intrinsics.cuh"
#include "../../util/reduction/serial_reduce.cuh"
#include "../../util/ns_wrapper.cuh"

#include "../../radix_sort/sort_utils.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {
namespace cta {


//---------------------------------------------------------------------
// Tuning policy types
//---------------------------------------------------------------------


/**
 * Upsweep CTA tuning policy
 */
template <
	int 							_RADIX_BITS,			// The number of radix bits, i.e., log2(bins)
	int 							_MIN_CTA_OCCUPANCY,		// The minimum CTA occupancy requested for this kernel per SM
	int 							_LOG_CTA_THREADS,		// The number of threads per CTA
	int 							_ELEMENTS_PER_THREAD,	// The number of elements to load per thread
	util::io::ld::CacheModifier 	_LOAD_MODIFIER,			// Load cache-modifier
	util::io::st::CacheModifier 	_STORE_MODIFIER,		// Store cache-modifier
	cudaSharedMemConfig				_SMEM_CONFIG,			// Shared memory bank size
	bool 							_EARLY_EXIT>			// Whether or not to short-circuit passes if the upsweep determines homogoneous digits in the current digit place
struct CtaUpsweepPassPolicy
{
	enum
	{
		RADIX_BITS					= _RADIX_BITS,
		MIN_CTA_OCCUPANCY  			= _MIN_CTA_OCCUPANCY,
		LOG_CTA_THREADS				= _LOG_CTA_THREADS,
		ELEMENTS_PER_THREAD  		= _ELEMENTS_PER_THREAD,
		EARLY_EXIT					= _EARLY_EXIT,

		CTA_THREADS					= 1 << LOG_CTA_THREADS,
		TILE_ELEMENTS				= CTA_THREADS * ELEMENTS_PER_THREAD,
	};

	static const util::io::ld::CacheModifier 	LOAD_MODIFIER 		= _LOAD_MODIFIER;
	static const util::io::st::CacheModifier 	STORE_MODIFIER 		= _STORE_MODIFIER;
	static const cudaSharedMemConfig			SMEM_CONFIG			= _SMEM_CONFIG;
};



//---------------------------------------------------------------------
// CTA-wide abstractions
//---------------------------------------------------------------------


/**
 * CTA-wide "upsweep" abstraction for computing radix digit histograms
 * over a range of input tiles.
 */
template <
	typename CtaUpsweepPassPolicy,
	typename KeyType,
	typename SizeT>
class CtaUpsweepPass
{
private:

	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	typedef typename KeyTraits<KeyType>::UnsignedBits UnsignedBits;

	// Integer type for digit counters (to be packed into words of PackedCounters)
	typedef unsigned char DigitCounter;

	// Integer type for packing DigitCounters into columns of shared memory banks
	typedef typename util::If<(CtaUpsweepPassPolicy::SMEM_CONFIG == cudaSharedMemBankSizeEightByte),
		unsigned long long,
		unsigned int>::Type PackedCounter;

	enum
	{
		RADIX_BITS					= CtaUpsweepPassPolicy::RADIX_BITS,
		RADIX_DIGITS 				= 1 << RADIX_BITS,

		LOG_CTA_THREADS 			= CtaUpsweepPassPolicy::LOG_CTA_THREADS,
		CTA_THREADS					= 1 << LOG_CTA_THREADS,

		LOG_WARP_THREADS 			= CUB_LOG_WARP_THREADS(__CUB_CUDA_ARCH__),
		WARP_THREADS				= 1 << LOG_WARP_THREADS,

		LOG_WARPS					= LOG_CTA_THREADS - LOG_WARP_THREADS,
		WARPS						= 1 << LOG_WARPS,

		KEYS_PER_THREAD  			= CtaUpsweepPassPolicy::ELEMENTS_PER_THREAD,

		TILE_ELEMENTS				= CTA_THREADS * KEYS_PER_THREAD,

		BYTES_PER_COUNTER			= sizeof(DigitCounter),
		LOG_BYTES_PER_COUNTER		= util::Log2<BYTES_PER_COUNTER>::VALUE,

		PACKING_RATIO				= sizeof(PackedCounter) / sizeof(DigitCounter),
		LOG_PACKING_RATIO			= util::Log2<PACKING_RATIO>::VALUE,

		LOG_COUNTER_LANES 			= CUB_MAX(0, RADIX_BITS - LOG_PACKING_RATIO),
		COUNTER_LANES 				= 1 << LOG_COUNTER_LANES,

		// To prevent counter overflow, we must periodically unpack and aggregate the
		// digit counters back into registers.  Each counter lane is assigned to a
		// warp for aggregation.

		LOG_LANES_PER_WARP			= CUB_MAX(0, LOG_COUNTER_LANES - LOG_WARPS),
		LANES_PER_WARP 				= 1 << LOG_LANES_PER_WARP,

		// Unroll tiles in batches without risk of counter overflow
		UNROLL_COUNT				= CUB_MIN(64, 255 / KEYS_PER_THREAD),
		UNROLLED_ELEMENTS 			= UNROLL_COUNT * TILE_ELEMENTS,
	};

public:

	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		union
		{
			DigitCounter 	digit_counters[COUNTER_LANES][CTA_THREADS][PACKING_RATIO];
			PackedCounter 	packed_counters[COUNTER_LANES][CTA_THREADS];
			SizeT 			digit_partials[RADIX_DIGITS][WARP_THREADS + 1];
		};
	};

private:

	//---------------------------------------------------------------------
	// Thread fields (aggregate state bundle)
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	SmemStorage 		&cta_smem_storage;

	// Thread-local counters for periodically aggregating composite-counter lanes
	SizeT 				local_counts[LANES_PER_WARP][PACKING_RATIO];

	// Input and output device pointers
	UnsignedBits		*d_keys_in;

	// The least-significant bit position of the current digit to extract
	unsigned int 		current_bit;



	//---------------------------------------------------------------------
	// Helper structure for templated iteration
	//---------------------------------------------------------------------

	// Iterate
	template <int COUNT, int MAX>
	struct Iterate
	{
		enum {
			HALF = (MAX / 2),
		};

		// BucketKeys
		static __device__ __forceinline__ void BucketKeys(
			CtaUpsweepPass &cta,
			UnsignedBits keys[KEYS_PER_THREAD])
		{
			cta.Bucket(keys[COUNT]);

			// Next
			Iterate<COUNT + 1, MAX>::BucketKeys(cta, keys);
		}

		// ProcessTiles
		static __device__ __forceinline__ void ProcessTiles(CtaUpsweepPass &cta, SizeT cta_offset)
		{
			// Next
			Iterate<1, HALF>::ProcessTiles(cta, cta_offset);
			Iterate<1, MAX - HALF>::ProcessTiles(cta, cta_offset + (HALF * TILE_ELEMENTS));
		}
	};

	// Terminate
	template <int MAX>
	struct Iterate<MAX, MAX>
	{
		// BucketKeys
		static __device__ __forceinline__ void BucketKeys(CtaUpsweepPass &cta, UnsignedBits keys[KEYS_PER_THREAD]) {}

		// ProcessTiles
		static __device__ __forceinline__ void ProcessTiles(CtaUpsweepPass &cta, SizeT cta_offset)
		{
			cta.ProcessFullTile(cta_offset);
		}
	};


	//---------------------------------------------------------------------
	// Utility methods
	//---------------------------------------------------------------------

	/**
	 * State bundle constructor
	 */
	__device__ __forceinline__ CtaUpsweepPass(
		SmemStorage		&cta_smem_storage,
		KeyType 		*d_keys_in,
		unsigned int 	current_bit) :
			cta_smem_storage(cta_smem_storage),
			d_keys_in(reinterpret_cast<UnsignedBits*>(d_keys_in)),
			current_bit(current_bit)
	{}


	/**
	 * Decode a key and increment corresponding smem digit counter
	 */
	__device__ __forceinline__ void Bucket(UnsignedBits key)
	{
		// Perform transform op
		UnsignedBits converted_key = KeyTraits<KeyType>::TwiddleIn(key);

		// Add in sub-counter offset
		UnsignedBits sub_counter = util::BFE(converted_key, current_bit, LOG_PACKING_RATIO);

		// Add in row offset
		UnsignedBits row_offset = util::BFE(converted_key, current_bit + LOG_PACKING_RATIO, LOG_COUNTER_LANES);

		// Increment counter
		cta_smem_storage.digit_counters[row_offset][threadIdx.x][sub_counter]++;

	}


	/**
	 * Reset composite counters
	 */
	__device__ __forceinline__ void ResetDigitCounters()
	{
		#pragma unroll
		for (int LANE = 0; LANE < COUNTER_LANES; LANE++)
		{
			cta_smem_storage.packed_counters[LANE][threadIdx.x] = 0;
		}
	}


	/**
	 * Reset the unpacked counters in each thread
	 */
	__device__ __forceinline__ void ResetUnpackedCounters()
	{
		#pragma unroll
		for (int LANE = 0; LANE < LANES_PER_WARP; LANE++)
		{
			#pragma unroll
			for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; UNPACKED_COUNTER++)
			{
				local_counts[LANE][UNPACKED_COUNTER] = 0;
			}
		}
	}


	/**
	 * Extracts and aggregates the digit counters for each counter lane
	 * owned by this warp
	 */
	__device__ __forceinline__ void UnpackDigitCounts()
	{
		unsigned int warp_id = threadIdx.x >> LOG_WARP_THREADS;
		unsigned int warp_tid = threadIdx.x & (WARP_THREADS - 1);

		if (warp_id < COUNTER_LANES)
		{
			#pragma unroll
			for (int LANE = 0; LANE < LANES_PER_WARP; LANE++)
			{
				const int COUNTER_LANE = LANE * WARPS;

				#pragma unroll
				for (int PACKED_COUNTER = 0; PACKED_COUNTER < CTA_THREADS; PACKED_COUNTER += WARP_THREADS)
				{
					#pragma unroll
					for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; UNPACKED_COUNTER++)
					{
						const int OFFSET = (((COUNTER_LANE * CTA_THREADS) + PACKED_COUNTER) * PACKING_RATIO) + UNPACKED_COUNTER;
						local_counts[LANE][UNPACKED_COUNTER] += cta_smem_storage.digit_counters[warp_id][warp_tid][OFFSET];
					}
				}
			}
		}
	}


	/**
	 * Places unpacked counters into smem for final digit reduction
	 */
	__device__ __forceinline__ void ReduceUnpackedCounts(SizeT &bin_count)
	{
		unsigned int warp_id = threadIdx.x >> LOG_WARP_THREADS;
		unsigned int warp_tid = threadIdx.x & (WARP_THREADS - 1);

		// Place unpacked digit counters in shared memory
		if (warp_id < COUNTER_LANES)
		{
			#pragma unroll
			for (int LANE = 0; LANE < LANES_PER_WARP; LANE++)
			{
				const int COUNTER_LANE = LANE * WARPS;
				int digit_row = (COUNTER_LANE + warp_id) << LOG_PACKING_RATIO;

				#pragma unroll
				for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; UNPACKED_COUNTER++)
				{
					cta_smem_storage.digit_partials[digit_row + UNPACKED_COUNTER][warp_tid]
						  = local_counts[LANE][UNPACKED_COUNTER];
				}
			}
		}

		__syncthreads();

		// Rake-reduce bin_count reductions
		if (threadIdx.x < RADIX_DIGITS)
		{
			bin_count = util::reduction::SerialReduce<WARP_THREADS>::Invoke(
				cta_smem_storage.digit_partials[threadIdx.x]);
		}
	}


	/**
	 * Processes a single, full tile
	 */
	__device__ __forceinline__ void ProcessFullTile(SizeT cta_offset)
	{
		// Tile of keys
		UnsignedBits keys[KEYS_PER_THREAD];

		#pragma unroll
		for (int LOAD = 0; LOAD < KEYS_PER_THREAD; LOAD++)
		{
			keys[LOAD] = d_keys_in[cta_offset + threadIdx.x + (LOAD * CTA_THREADS)];
		}

		// Bucket tile of keys
		Iterate<0, KEYS_PER_THREAD>::BucketKeys(*this, keys);
	}


	/**
	 * Processes a single load (may have some threads masked off)
	 */
	__device__ __forceinline__ void ProcessPartialTile(
		SizeT cta_offset,
		const SizeT &num_elements)
	{
		// Process partial tile if necessary using single loads
		cta_offset += threadIdx.x;
		while (cta_offset < num_elements)
		{
			// Load and bucket key
			UnsignedBits key = d_keys_in[cta_offset];
			Bucket(key);
			cta_offset += CTA_THREADS;
		}
	}

public:

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Compute radix digit histograms over a range of input tiles.
	 */
	static __device__ __forceinline__ void UpsweepPass(
		SmemStorage 	&cta_smem_storage,
		KeyType 		*d_keys_in,
		unsigned int 	current_bit,
		const SizeT 	&num_elements,
		SizeT 			&bin_count)				// The digit count for tid'th bin (output param, valid in the first RADIX_DIGITS threads)
	{
		// Construct state bundle
		CtaUpsweepPass cta(cta_smem_storage, d_keys_in, current_bit);

		// Reset digit counters in smem and unpacked counters in registers
		cta.ResetDigitCounters();
		cta.ResetUnpackedCounters();

		// Unroll batches of full tiles
		SizeT cta_offset = 0;
		while (cta_offset + UNROLLED_ELEMENTS <= num_elements)
		{
			Iterate<0, UNROLL_COUNT>::ProcessTiles(cta, cta_offset);
			cta_offset += UNROLLED_ELEMENTS;

			__syncthreads();

			// Aggregate back into local_count registers to prevent overflow
			cta.UnpackDigitCounts();

			__syncthreads();

			// Reset composite counters in lanes
			cta.ResetDigitCounters();
		}

		// Unroll single full tiles
		while (cta_offset + TILE_ELEMENTS <= num_elements)
		{
			cta.ProcessFullTile(cta_offset);
			cta_offset += TILE_ELEMENTS;
		}

		// Process partial tile if necessary
		cta.ProcessPartialTile(
			cta_offset,
			num_elements);

		__syncthreads();

		// Aggregate back into local_count registers
		cta.UnpackDigitCounts();

		__syncthreads();

		// Final raking reduction of counts by bin
		cta.ReduceUnpackedCounts(bin_count);
	}

};


} // namespace cta
} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
