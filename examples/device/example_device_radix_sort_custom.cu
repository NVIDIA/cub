/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/device/device_radix_sort.cuh>

#include <bitset>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct custom_t
{
  std::uint16_t i;
  float f;
};

struct custom_decomposer_t
{
  __host__ __device__ ::cuda::std::tuple<std::uint16_t&, float&> operator()(custom_t &key) const
  {
    return {key.i, key.f};
  }
};

void print_intro()
{
  std::cout << "This example illustrates use of radix sort with custom type.\n";
  std::cout << "Let's define a simple structure of the following form:\n\n";
  std::cout << "\tstruct custom_t {\n";
  std::cout << "\t  std::uint32_t i;\n";
  std::cout << "\t  float f;\n";
  std::cout << "\t};\n\n";
  std::cout << "The `i` field is already stored in the bit-lexicographical order.\n";
  std::cout << "The `f` field, however, isn't. Therefore, to feed this structure \n";
  std::cout << "into the radix sort, we have to convert `f` into bit ordered representation.\n";
  std::cout << "The `custom_t{65535, -4.2f}` has the following binary representation:\n\n";

  auto print_segment = [](std::string msg, std::size_t segment_size, char filler = '-') {
    std::string spaces((segment_size - msg.size()) / 2 - 1, filler);
    std::cout << '<' << spaces << msg << spaces << '>';
  };

  std::cout << '\t';
  print_segment(" `.f` ", 32);
  print_segment(" padding -", 16);
  print_segment(" `.s` ", 16);
  std::cout << '\n';

  std::cout << "\ts";
  print_segment(" exp. ", 8);
  print_segment(" mantissa -", 23);
  print_segment(" padding -", 16);
  print_segment(" short -", 16);
  std::cout << '\n';

  custom_t the_answer{65535, -4.2f};
  std::cout << '\t' << std::bitset<64>{reinterpret_cast<std::uint64_t &>(the_answer)};
  std::cout << "\n\t";
  print_segment(" <----  higher bits  /  lower bits  ----> ", 64, ' ');
  std::cout << "\n\n";

  std::cout << "Let's say we are trying to compare l={42, -4.2f} with g={42, 4.2f}:\n";

  std::cout << '\t';
  print_segment(" `.f` ", 32);
  print_segment(" `.i` ", 32);
  std::cout << '\n';

  custom_t l{42, -4.2f};
  custom_t g{42, 4.2f};
  std::cout << "l:\t" << std::bitset<64>{reinterpret_cast<std::uint64_t &>(l)} << '\n';
  std::cout << "g:\t" << std::bitset<64>{reinterpret_cast<std::uint64_t &>(g)} << "\n\n";

  std::cout << "As you can see, `l` key happened to be larger in the bit-lexicographicl order.\n";
  std::cout << "Since there's no reflection in C++, we can't expect the type and convert \n";
  std::cout << "each field into the bit-lexicographicl order. You can tell CUB how to do that\n";
  std::cout << "by specializing cub::RadixTraits for the `custom_t`:\n\n";

  std::cout << "\tstruct custom_bit_conversation_policy_t {\n";
  std::cout << "\t  using fp_traits                 = cub::RadixTraits<float>;\n";
  std::cout << "\t  using fp_bit_ordered            = fp_traits::bit_ordered_type;\n";
  std::cout << "\t  using fp_bit_ordered_conversion = fp_traits::bit_ordered_conversion_policy;\n";
  std::cout << "\t\n";
  std::cout << "\t  static __host__ __device__ std::uint64_t to_bit_ordered(std::uint64_t val) {\n";
  std::cout
    << "\t    return embed_fp(val, fp_bit_ordered_conversion::to_bit_ordered(extract_fp(val)));\n";
  std::cout << "\t  }\n\n";
  std::cout
    << "\t  static __host__ __device__ std::uint64_t from_bit_ordered(std::uint64_t val) {\n";
  std::cout << "\t    return embed_fp(val, "
               "fp_bit_ordered_conversion::from_bit_ordered(extract_fp(val)));\n";
  std::cout << "\t  }\n";
  std::cout << "\t};\n\n";

  std::cout << "After conversion we have:\n\n";

  std::cout << '\t';
  print_segment(" `.f` ", 32);
  print_segment(" `.i` ", 32);
  std::cout << '\n';

  std::cout << "\ts";
  print_segment(" exp. ", 8);
  print_segment(" mantissa -", 23);
  print_segment(" unsigned integer ", 32);
  std::cout << '\n';
}

void sort()
{
  int n = 2;
  thrust::device_vector<custom_t> in(n);
  thrust::device_vector<custom_t> out(n);

  in[0] = custom_t{ 0, 1 };
  in[1] = custom_t{ 0, 0 };

  std::uint8_t *d_temp_storage{};
  std::size_t temp_storage_size{};

  // example-begin keys-only
  const custom_t *d_in = thrust::raw_pointer_cast(in.data());
  custom_t *d_out = thrust::raw_pointer_cast(out.data());

  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_size, 
                                 d_in, d_out, n, custom_decomposer_t{});

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_size);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_size, 
                                 d_in, d_out, n, custom_decomposer_t{});
  // example-end keys-only

  thrust::host_vector<custom_t> result = out;
  std::cout << "{ " << result[0].i << " " << result[0].f << " }" << std::endl; 
  std::cout << "{ " << result[1].i << " " << result[1].f << " }" << std::endl; 
}

template <class T>
void print(T)
{
  std::cout << __PRETTY_FUNCTION__ << std::endl;
}

void print(custom_t aggregate)
{
  std::cout << std::bitset<64>(reinterpret_cast<std::uint64_t &>(aggregate)) << std::endl;
}

void print(std::uint32_t aggregate) { std::cout << std::bitset<32>(aggregate) << std::endl; }

int main()
{
  /*
  custom_t key{4, 2.0f};

  ::cub::detail::max_raw_binary_key(custom_decomposer_t{}, key);
  print(key);

  ::cub::detail::min_raw_binary_key(custom_decomposer_t{}, key);
  print(key);

  ::cub::detail::to_bit_ordered(custom_decomposer_t{}, key);
  print(key);

  ::cub::detail::from_bit_ordered(custom_decomposer_t{}, key);
  print(key);

  ::cub::detail::inverse(custom_decomposer_t{}, key.i);
  print(key);

  const int end_bit = ::cub::detail::default_end_bit(custom_decomposer_t{}, key);
  std::cout << "end bit = " << end_bit << std::endl;

  for (int begin_bit = 16; begin_bit < end_bit; begin_bit += 4) {
    std::uint32_t bits = ::cub::detail::custom_digit_extractor_t<custom_decomposer_t>(custom_decomposer_t{}, begin_bit, 4).Digit(key);
    std::cout << begin_bit << " -> " << std::bitset<4>(bits) << std::endl;
  }
  */

  // sort();
  print_intro();

  return 0;
}
