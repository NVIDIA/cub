/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/util_ptx.cuh>
#include <test_util.h>

bool IsLaneInvolved(unsigned int member_mask, unsigned int lane)
{
  return member_mask & (1 << lane);
}

template <int LOGICAL_WARP_THREADS>
void Test()
{
  constexpr bool is_pow_of_two = cub::PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE;
  constexpr unsigned int warp_threads = 32;
  constexpr unsigned int warps = warp_threads / LOGICAL_WARP_THREADS;

  for (unsigned int warp_id = 0; warp_id < warps; warp_id++)
  {
    const unsigned int warp_mask =
      cub::WarpMask<LOGICAL_WARP_THREADS, 860>(warp_id);

    const unsigned int warp_begin = LOGICAL_WARP_THREADS * warp_id;
    const unsigned int warp_end = warp_begin + LOGICAL_WARP_THREADS;

    if (is_pow_of_two)
    {
      for (unsigned int prev_warp_lane = 0;
           prev_warp_lane < warp_begin;
           prev_warp_lane++)
      {
        AssertEquals(IsLaneInvolved(warp_mask, prev_warp_lane), false);
      }

      for (unsigned int warp_lane = warp_begin;
           warp_lane < warp_end;
           warp_lane++)
      {
        AssertEquals(IsLaneInvolved(warp_mask, warp_lane), true);
      }

      for (unsigned int post_warp_lane = warp_end;
           post_warp_lane < warp_threads;
           post_warp_lane++)
      {
        AssertEquals(IsLaneInvolved(warp_mask, post_warp_lane), false);
      }
    }
    else
    {
      for (unsigned int warp_lane = 0;
           warp_lane < LOGICAL_WARP_THREADS;
           warp_lane++)
      {
        AssertEquals(IsLaneInvolved(warp_mask, warp_lane), true);
      }

      for (unsigned int warp_lane = LOGICAL_WARP_THREADS;
           warp_lane < warp_threads;
           warp_lane++)
      {
        AssertEquals(IsLaneInvolved(warp_mask, warp_lane), false);
      }
    }
  }
}


void TestPowersOfTwo()
{
  Test<1>();
  Test<2>();
  Test<4>();
  Test<8>();
  Test<16>();
  Test<32>();
}


void TestNonPowersOfTwo()
{
  Test<3>();
  Test<5>();
  Test<6>();
  Test<7>();

  Test<9>();
  Test<10>();
  Test<11>();
  Test<12>();
  Test<13>();
  Test<14>();
  Test<15>();

  Test<17>();
  Test<18>();
  Test<19>();
  Test<20>();
  Test<21>();
  Test<22>();
  Test<23>();
  Test<24>();
  Test<25>();
  Test<26>();
  Test<27>();
  Test<28>();
  Test<29>();
  Test<30>();
  Test<31>();
}


int main()
{
  TestPowersOfTwo();
  TestNonPowersOfTwo();
}
