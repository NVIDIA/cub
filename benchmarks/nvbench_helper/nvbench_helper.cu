#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/tabulate.h>

#include <cstdint>
#include <random>

#include <curand.h>
#include <nvbench_helper.cuh>

class generator_t
{
public:
  generator_t();
  ~generator_t();

  template <typename T>
  void operator()(seed_t seed,
                  thrust::device_vector<T> &data,
                  bit_entropy entropy,
                  T min = std::numeric_limits<T>::lowest(),
                  T max = std::numeric_limits<T>::max());

  double *distribution();
  curandGenerator_t &gen() { return m_gen; }

  double *prepare_random_generator(seed_t seed, std::size_t num_items);

private:
  curandGenerator_t m_gen;
  thrust::device_vector<double> m_distribution;
};

template <typename T>
struct random_to_item_t
{
  double m_min;
  double m_max;

  __host__ __device__ random_to_item_t(T min, T max)
      : m_min(static_cast<double>(min))
      , m_max(static_cast<double>(max))
  {}

  __host__ __device__ T operator()(double random_value)
  {
    return static_cast<T>((m_max - m_min) * random_value + m_min);
  }
};

generator_t::generator_t() { curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_DEFAULT); }

generator_t::~generator_t() { curandDestroyGenerator(m_gen); }

double *generator_t::distribution() { return thrust::raw_pointer_cast(m_distribution.data()); }

double *generator_t::prepare_random_generator(seed_t seed, std::size_t num_items)
{
  curandSetPseudoRandomGeneratorSeed(m_gen, seed.get());

  m_distribution.resize(num_items);
  curandGenerateUniformDouble(m_gen, this->distribution(), num_items);

  return this->distribution();
}

template <class T>
__global__ void and_kernel(T *d_in, T *d_tmp, std::size_t n)
{
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    d_in[i] = d_in[i] & d_tmp[i];
  }
}

__global__ void and_kernel(float *d_in, float *d_tmp, std::size_t n)
{
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    std::uint32_t result = reinterpret_cast<std::uint32_t &>(d_in[i]) &
                           reinterpret_cast<std::uint32_t &>(d_tmp[i]);
    d_in[i] = reinterpret_cast<float &>(result);
  }
}

__global__ void and_kernel(double *d_in, double *d_tmp, std::size_t n)
{
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    std::uint64_t result = reinterpret_cast<std::uint64_t &>(d_in[i]) &
                           reinterpret_cast<std::uint64_t &>(d_tmp[i]);
    d_in[i] = reinterpret_cast<double &>(result);
  }
}

__global__ void and_kernel(complex *d_in, complex *d_tmp, std::size_t n)
{
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    double in_real = d_in[i].real();
    double in_imag = d_in[i].imag();

    double tmp_real = d_tmp[i].real();
    double tmp_imag = d_tmp[i].imag();

    std::uint64_t result_real = reinterpret_cast<std::uint64_t &>(in_real) &
                                reinterpret_cast<std::uint64_t &>(tmp_real);

    std::uint64_t result_imag = reinterpret_cast<std::uint64_t &>(in_imag) &
                                reinterpret_cast<std::uint64_t &>(tmp_imag);

    d_in[i].real(reinterpret_cast<double &>(result_real));
    d_in[i].imag(reinterpret_cast<double &>(result_imag));
  }
}

__global__ void set_real_kernel(complex *d_in, complex min, complex max, double *d_tmp, std::size_t n)
{
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    d_in[i].real(random_to_item_t<double>{min.real(), max.real()}(d_tmp[i]));
  }
}

__global__ void set_imag_kernel(complex *d_in, complex min, complex max, double *d_tmp, std::size_t n)
{
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    d_in[i].imag(random_to_item_t<double>{min.imag(), max.imag()}(d_tmp[i]));
  }
}

template <class T>
void generator_t::operator()(seed_t seed,
                             thrust::device_vector<T> &data,
                             bit_entropy entropy,
                             T min,
                             T max)
{
  switch (entropy)
  {
    case bit_entropy::_1_000: {
      prepare_random_generator(seed, data.size());

      thrust::transform(m_distribution.begin(),
                        m_distribution.end(),
                        data.begin(),
                        random_to_item_t<T>(min, max));
      return;
    }
    case bit_entropy::_0_000: {
      std::mt19937 rng;
      rng.seed(static_cast<std::mt19937::result_type>(seed.get()));
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      T random_value = random_to_item_t<T>(min, max)(dist(rng));
      thrust::fill(data.begin(), data.end(), random_value);
      return;
    }
    default: {
      prepare_random_generator(seed, data.size());

      thrust::transform(m_distribution.begin(),
                        m_distribution.end(),
                        data.begin(),
                        random_to_item_t<T>(min, max));

      const int number_of_steps = static_cast<int>(entropy);
      thrust::device_vector<T> tmp(data.size());
      const int threads_in_block = 256;
      const int blocks_in_grid   = (data.size() + threads_in_block - 1) / threads_in_block;

      for (int i = 0; i < number_of_steps; i++, ++seed)
      {
        (*this)(seed, tmp, bit_entropy::_1_000, min, max);
        and_kernel<<<blocks_in_grid, threads_in_block>>>(thrust::raw_pointer_cast(data.data()),
                                                         thrust::raw_pointer_cast(tmp.data()),
                                                         data.size());
        cudaStreamSynchronize(0);
      }
      return;
    }
  };
}

template <>
void generator_t::operator()(seed_t seed,
                             thrust::device_vector<complex> &data,
                             bit_entropy entropy,
                             complex min,
                             complex max)
{
  const int threads_in_block = 256;
  const int blocks_in_grid   = (data.size() + threads_in_block - 1) / threads_in_block;

  switch (entropy)
  {
    case bit_entropy::_1_000: {
      prepare_random_generator(seed, data.size()); ++seed;
      set_real_kernel<<<blocks_in_grid, threads_in_block>>>(thrust::raw_pointer_cast(data.data()),
                                                            min,
                                                            max,
                                                            thrust::raw_pointer_cast(m_distribution.data()),
                                                            data.size());
      prepare_random_generator(seed, data.size()); ++seed;
      set_imag_kernel<<<blocks_in_grid, threads_in_block>>>(thrust::raw_pointer_cast(data.data()),
                                                            min,
                                                            max,
                                                            thrust::raw_pointer_cast(m_distribution.data()),
                                                            data.size());
      return;
    }
    case bit_entropy::_0_000: {
      std::mt19937 rng;
      rng.seed(static_cast<std::mt19937::result_type>(seed.get()));
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      double random_imag = random_to_item_t<double>(min.imag(), max.imag())(dist(rng));
      double random_real = random_to_item_t<double>(min.imag(), max.imag())(dist(rng));
      complex random_value(random_real, random_imag);
      thrust::fill(data.begin(), data.end(), random_value);
      return;
    }
    default: {
      prepare_random_generator(seed, data.size());

      prepare_random_generator(seed, data.size()); ++seed;
      set_real_kernel<<<blocks_in_grid, threads_in_block>>>(thrust::raw_pointer_cast(data.data()),
                                                            min,
                                                            max,
                                                            thrust::raw_pointer_cast(m_distribution.data()),
                                                            data.size());
      prepare_random_generator(seed, data.size()); ++seed;
      set_imag_kernel<<<blocks_in_grid, threads_in_block>>>(thrust::raw_pointer_cast(data.data()),
                                                            min,
                                                            max,
                                                            thrust::raw_pointer_cast(m_distribution.data()),
                                                            data.size());

      const int number_of_steps = static_cast<int>(entropy);
      thrust::device_vector<complex> tmp(data.size());

      for (int i = 0; i < number_of_steps; i++, ++seed)
      {
        (*this)(seed, tmp, bit_entropy::_1_000, min, max);
        and_kernel<<<blocks_in_grid, threads_in_block>>>(thrust::raw_pointer_cast(data.data()),
                                                         thrust::raw_pointer_cast(tmp.data()),
                                                         data.size());
        cudaStreamSynchronize(0);
      }
      return;
    }
  };
}

struct random_to_probability_t
{
  double m_probability;

  __host__ __device__ bool operator()(double random_value)
  {
    return random_value < m_probability;
  }
};

template <>
void generator_t::operator()(seed_t seed,
                             thrust::device_vector<bool> &data,
                             bit_entropy entropy,
                             bool /* min */,
                             bool /* max */)
{
  if (entropy == bit_entropy::_0_000)
  {
    thrust::fill(data.begin(), data.end(), false);
  }
  else if (entropy == bit_entropy::_1_000)
  {
    thrust::fill(data.begin(), data.end(), true);
  }
  else
  {
    prepare_random_generator(seed, data.size());
    thrust::transform(m_distribution.begin(),
                      m_distribution.end(),
                      data.begin(),
                      random_to_probability_t{entropy_to_probability(entropy)});
  }
}

template <typename T>
void gen(seed_t seed, thrust::device_vector<T> &data, bit_entropy entropy, T min, T max)
{
  generator_t{}(seed, data, entropy, min, max);
}

#define INSTANTIATE_RND(TYPE)                                                                      \
  template void gen<TYPE>(seed_t,                                                                  \
                          thrust::device_vector<TYPE> & data,                                      \
                          bit_entropy,                                                             \
                          TYPE min,                                                                \
                          TYPE max)

#define INSTANTIATE(TYPE) INSTANTIATE_RND(TYPE);

INSTANTIATE(bool);

INSTANTIATE(uint8_t);
INSTANTIATE(uint16_t);
INSTANTIATE(uint32_t);
INSTANTIATE(uint64_t);
INSTANTIATE(uint128_t);

INSTANTIATE(int8_t);
INSTANTIATE(int16_t);
INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
INSTANTIATE(int128_t);

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(complex);
