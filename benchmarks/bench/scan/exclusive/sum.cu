// %RANGE% TUNE_ITEMS ipt 7:24:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% CUB_DETAIL_L2_BACKOFF_NS l2b 0:1200:5
// %RANGE% CUB_DETAIL_L2_WRITE_LATENCY_NS l2w 0:1200:5

#include <nvbench_helper.cuh>

using op_t = cub::Sum;
#include "base.cuh"
