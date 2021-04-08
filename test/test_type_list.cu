/******************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/detail/type_list.cuh>

#include <cuda/std/type_traits>

#include "test_util.h"

using cub::detail::type_list;
using namespace cub::detail::tl;

struct T1
{};
struct T2
{};
struct T3
{};
struct T4
{};

void test_type_list()
{
  StaticAssertSame((type_list<>), (type_list<>));
  StaticAssertSame((type_list<T1>), (type_list<T1>));
  StaticAssertSame((type_list<T1, T2>), (type_list<T1, T2>));
  StaticAssertDiff((type_list<T1, T2>), (type_list<T2, T1>));
}

void test_contains()
{
  StaticAssertEquals((contains<type_list<T1, T2, T3>, T1>::value), (true));
  StaticAssertEquals((contains<type_list<T1, T2, T3>, T2>::value), (true));
  StaticAssertEquals((contains<type_list<T1, T2, T3>, T3>::value), (true));
  StaticAssertEquals((contains<type_list<T1, T2, T3>, T4>::value), (false));
  StaticAssertEquals((contains<type_list<>, T4>::value), (false));
}

void test_prepend()
{
  StaticAssertSame((typename prepend<type_list<>, T1>::type), (type_list<T1>));
  StaticAssertSame((typename prepend<type_list<T2>, T1>::type), (type_list<T1, T2>));
  StaticAssertSame((typename prepend<type_list<T1>, T2>::type), (type_list<T2, T1>));
  StaticAssertSame((typename prepend<type_list<T1, T2>, T3>::type), (type_list<T3, T1, T2>));
  StaticAssertSame((typename prepend<type_list<T1, T2, T3>, T1>::type), (type_list<T1, T1, T2, T3>));
  StaticAssertSame((typename prepend<type_list<T1, T2, T3>, T2>::type), (type_list<T2, T1, T2, T3>));
  StaticAssertSame((typename prepend<type_list<T1, T2, T3>, T3>::type), (type_list<T3, T1, T2, T3>));
  StaticAssertSame((typename prepend<type_list<T1, T2, T3>, T4>::type), (type_list<T4, T1, T2, T3>));
}

void test_append()
{
  StaticAssertSame((typename append<type_list<>, T1>::type), (type_list<T1>));
  StaticAssertSame((typename append<type_list<T2>, T1>::type), (type_list<T2, T1>));
  StaticAssertSame((typename append<type_list<T1>, T2>::type), (type_list<T1, T2>));
  StaticAssertSame((typename append<type_list<T1, T2>, T3>::type), (type_list<T1, T2, T3>));
  StaticAssertSame((typename append<type_list<T1, T2, T3>, T1>::type), (type_list<T1, T2, T3, T1>));
  StaticAssertSame((typename append<type_list<T1, T2, T3>, T2>::type), (type_list<T1, T2, T3, T2>));
  StaticAssertSame((typename append<type_list<T1, T2, T3>, T3>::type), (type_list<T1, T2, T3, T3>));
  StaticAssertSame((typename append<type_list<T1, T2, T3>, T4>::type), (type_list<T1, T2, T3, T4>));
}

int main()
{
  test_type_list();
  test_contains();
  test_prepend();
  test_append();
}
