#!/usr/bin/env python3

import cub.bench 


def workload_header(ct_workload_space, rt_workload_space):
    for ct_workload in ct_workload_space:
        for rt_workload in rt_workload_space:
            workload_point = ct_workload + rt_workload
            return ", ".join([x.split('=')[0] for x in workload_point])


def workload_entry(ct_workload, rt_workload):
    workload_point = ct_workload + rt_workload
    return ", ".join([x.split('=')[1] for x in workload_point])


class VerifySeeker:
    def __init__(self):
        self.estimator = cub.bench.MedianCenterEstimator()

    def __call__(self, algname, ct_workload_space, rt_workload_space):
        label = 'trp_0.ld_0.ipt_20.tpb_448.l2b_485.l2w_710'
        variant_point = cub.bench.Config().label_to_variant_point(algname, label)

        print("{}, MinS, MedianS, MaxS".format(workload_header(ct_workload_space, rt_workload_space)))
        for ct_workload in ct_workload_space:
            bench = cub.bench.Bench(algname, variant_point, list(ct_workload))
            if bench.build():
                base = bench.get_base()
                for rt_workload in rt_workload_space:
                    workload_point = ct_workload + rt_workload
                    base_samples, base_elapsed = base.do_run(workload_point, None)
                    variant_samples, _ = bench.do_run(workload_point, base_elapsed * 10)
                    min_speedup = min(base_samples) / min(variant_samples)
                    median_speedup = self.estimator(base_samples) / self.estimator(variant_samples)
                    max_speedup = max(base_samples) / max(variant_samples)
                    point_str = workload_entry(ct_workload, rt_workload)
                    print("{}, {}, {}, {}".format(point_str, min_speedup, median_speedup, max_speedup))


def main():
    cub.bench.search(VerifySeeker())


if __name__ == "__main__":
    main()
