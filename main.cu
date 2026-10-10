/*
GPU: NVIDIA GeForce RTX 5070 Ti
SM count: 70
cuBLAS Version: 130800

GEMM Size:
M=8192 N=8192 K=8192

Average Time: 30.569 ms
FLOPs: 1.100 TFLOP
cuBLAS FP32 Performance: 35.968 TFLOPS


GEMM Size: 
M=2048 N=2048 K=2048
Average Time: 0.569 ms 
FLOPs: 0.017 TFLOP 
cuBLAS FP32 Performance: 30.181 TFLOPS

Done.
*/

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <iomanip>


#define CHECK_CUDA(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if(err != cudaSuccess){                                   \
        std::cerr << "CUDA Error: "                          \
                  << cudaGetErrorString(err)                 \
                  << std::endl;                              \
        exit(EXIT_FAILURE);                                  \
    }                                                         \
} while(0)



#define CHECK_CUBLAS(call)                                    \
do {                                                          \
    cublasStatus_t status = call;                             \
    if(status != CUBLAS_STATUS_SUCCESS){                      \
        std::cerr << "cuBLAS Error" << std::endl;             \
        exit(EXIT_FAILURE);                                  \
    }                                                         \
} while(0)



int main()
{

    // ============================
    // GPU 信息
    // ============================

    cudaDeviceProp prop;

    CHECK_CUDA(
        cudaGetDeviceProperties(&prop,0)
    );


    std::cout
        << "GPU: "
        << prop.name
        << std::endl;


    std::cout
        << "SM count: "
        << prop.multiProcessorCount
        << std::endl;



    // ============================
    // cuBLAS handle
    // ============================

    cublasHandle_t handle;

    CHECK_CUBLAS(
        cublasCreate(&handle)
    );


    int cublasVersion;

    CHECK_CUBLAS(
        cublasGetVersion(
            handle,
            &cublasVersion
        )
    );


    std::cout
        << "cuBLAS Version: "
        << cublasVersion
        << std::endl;



    // ============================
    // GEMM 参数
    // ============================

    int M = 2048;
    int N = 2048;
    int K = 2048;


    std::cout
        << "\nGEMM Size:"
        << std::endl;


    std::cout
        << "M="
        << M
        << " N="
        << N
        << " K="
        << K
        << std::endl;



    float *A;
    float *B;
    float *C;


    CHECK_CUDA(
        cudaMalloc(
            &A,
            M*K*sizeof(float)
        )
    );


    CHECK_CUDA(
        cudaMalloc(
            &B,
            K*N*sizeof(float)
        )
    );


    CHECK_CUDA(
        cudaMalloc(
            &C,
            M*N*sizeof(float)
        )
    );



    // 初始化
    CHECK_CUDA(
        cudaMemset(
            A,
            1,
            M*K*sizeof(float)
        )
    );


    CHECK_CUDA(
        cudaMemset(
            B,
            1,
            K*N*sizeof(float)
        )
    );


    CHECK_CUDA(
        cudaMemset(
            C,
            0,
            M*N*sizeof(float)
        )
    );



    float alpha = 1.0f;
    float beta  = 0.0f;



    // ============================
    // Warmup
    // ============================

    for(int i=0;i<10;i++)
    {

        CHECK_CUBLAS(
            cublasSgemm(

                handle,

                CUBLAS_OP_N,
                CUBLAS_OP_N,


                N,
                M,
                K,


                &alpha,

                B,
                N,

                A,
                K,

                &beta,

                C,
                N
            )
        );
    }


    CHECK_CUDA(cudaDeviceSynchronize());



    // ============================
    // Benchmark
    // ============================


    cudaEvent_t start, stop;


    cudaEventCreate(&start);
    cudaEventCreate(&stop);



    int repeat = 50;


    cudaEventRecord(start);


    for(int i=0;i<repeat;i++)
    {

        CHECK_CUBLAS(
            cublasSgemm(

                handle,

                CUBLAS_OP_N,
                CUBLAS_OP_N,


                N,
                M,
                K,


                &alpha,

                B,
                N,

                A,
                K,

                &beta,

                C,
                N
            )
        );

    }


    cudaEventRecord(stop);


    cudaEventSynchronize(stop);



    float total_ms;


    cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    );


    float avg_ms =
        total_ms / repeat;



    // ============================
    // TFLOPS
    // ============================

    double FLOPs =
        2.0 *
        M *
        N *
        K;



    double TFLOPS =
        FLOPs /
        (avg_ms / 1000.0) /
        1e12;



    std::cout
        << std::fixed
        << std::setprecision(3);



    std::cout
        << "\nAverage Time: "
        << avg_ms
        << " ms"
        << std::endl;



    std::cout
        << "FLOPs: "
        << FLOPs / 1e12
        << " TFLOP"
        << std::endl;



    std::cout
        << "cuBLAS FP32 Performance: "
        << TFLOPS
        << " TFLOPS"
        << std::endl;



    std::cout
        << "\nDone."
        << std::endl;



    cudaFree(A);
    cudaFree(B);
    cudaFree(C);


    cublasDestroy(handle);


    return 0;
}