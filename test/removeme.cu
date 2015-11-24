

#include <stdio.h>


#ifndef __CUDA_ARCH__
    #define CUB_PTX_ARCH 0
#else
    #define CUB_PTX_ARCH __CUDA_ARCH__
#endif


template <bool IF, typename ThenType, typename ElseType>
struct If
{
    typedef ThenType Type;
};

template <typename ThenType, typename ElseType>
struct If<false, ThenType, ElseType>
{
    typedef ElseType Type;
};


template <typename T>
__global__ void EmptyKernel(void) { }


template <int VERSION, typename PolicyUmbrellaT, typename PrevPolicyUmbrellaT>
struct PolicyDelegate
{
    // The PolicyUmbrella for the active compiler pass
    typedef typename If<(CUB_PTX_ARCH < VERSION),
            PrevPolicyUmbrellaT::ActivePolicyUmbrella,
            PolicyUmbrellaT>::Type
        ActivePolicyUmbrella;

    template <typename FunctorT>
    static void Invoke(int ptx_version, FunctorT op)
    {
        if (ptx_version < VERSION) {
            PrevPolicyUmbrellaT::Invoke(ptx_version, op);
            return;
        }
        op.template Invoke<PolicyUmbrellaT>();
    }
};

template <int VERSION, typename PolicyUmbrellaT>
struct PolicyDelegate<VERSION, PolicyUmbrellaT, PolicyUmbrellaT>
{
    // The PolicyUmbrella for the active compiler pass
    typedef PolicyUmbrellaT ActivePolicyUmbrella;

    template <typename FunctorT>
    static void Invoke(int ptx_version, FunctorT op) {
        op.template Invoke<PolicyUmbrellaT>();
    }
};





template <int _FOO, int _BAR = _FOO>
struct AgentPolicy
{
    enum {
        FOO = _FOO,
        BAR = _BAR,
    };
};


template <typename PolicyUmbrellaT>
__global__ void Kernel()
{
    typedef typename PolicyUmbrellaT::ActivePolicyUmbrella::PtxAgentPolicy PtxAgentPolicy;

    printf("PolicyUmbrellaT::ActivePtxPolicy::FOO = %d\n", PtxAgentPolicy::FOO);
}


struct Policy100 : PolicyDelegate<100, Policy100, Policy100>
{
    typedef AgentPolicy<10>             PtxAgentPolicy;
};

struct Policy300 : PolicyDelegate<300, Policy300, Policy100>
{
    typedef AgentPolicy<30>             PtxAgentPolicy;
};

struct MaxPolicy : PolicyDelegate<350, MaxPolicy, Policy300>
{
    typedef AgentPolicy<35>             PtxAgentPolicy;
};


struct InvokeFunctor
{
    int param1, param2;

    InvokeFunctor (int param1, int param2) : param1(param1), param2(param2) {}

    template <typename ActivePolicyUmbrellaT>
    void Invoke()
    {
        printf("Invoke FOO: %d\n", ActivePolicyUmbrellaT::PtxAgentPolicy::FOO);

        Kernel<MaxPolicy><<<1,1>>>();

        cudaDeviceSynchronize();
    }
};


int main()
{
    int ptx_version;
    cudaFuncAttributes empty_kernel_attrs;
    cudaFuncGetAttributes(&empty_kernel_attrs, EmptyKernel<void>);
    ptx_version = empty_kernel_attrs.ptxVersion * 10;

    printf("Ptx version %d\n", ptx_version);

    InvokeFunctor invoke_functor(1, 2);

    MaxPolicy::Invoke(ptx_version, invoke_functor);

    return 0;
}
