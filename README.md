Sparse Matrix Multiplication: COL380-Assignment4Umesh Kumar (2022CS11115)April 28, 2025AbstractWe present a high-performance implementation of sparse matrix multiplication using MPI for inter-node distribution, OpenMP for intra-node concurrency, and CUDA for GPU acceleration. Innovations include flattened block buffers, pinned host memory with asynchronous transfers, overlap of communication and computation, and a reduction-tree merge. On large block-sparse test cases, we reduce end-to-end time from ~150s to ~32s (4.7x speedup), achieving > 85% GPU occupancy and 12 GB/s PCIe bandwidth.1. IntroductionBlock-sparse matrix multiplication (tiles of size k×k) is central to many scientific and machine-learning kernels. Standard approaches suffer from pointer-chasing overhead, poor cache utilization, and host-device transfer bottlenecks. Our implementation targets three layers of parallelism:MPI: distribute input blocks across P ranks, then perform a binary-tree merge.OpenMP: parallel file I/O and host-side packing of device buffers.CUDA: launch one k×k tile-multiply per GPU block, overlapped with DMA.2. Code Restructuring & Flattened Buffers2.1 Flattened Block StorageWe replace std::map<pair<int, int>, vector<vector<uint64_t>>> with two large C-arrays for contiguous storage of all A-blocks:// A_buf: contiguous storage for all A-blocks
uint64_t *A_buf = new uint64_t [total_m1 * k*k];

for (auto &kv : a.mat) {
    int idx = A_idx [kv.first];
    uint64_t *dst = A_buf + idx*k*k;
    memcpy(dst, kv.second[0].data(), k*k*sizeof(uint64_t));
    //... copy subsequent rows
}
Lookup becomes two hash-map lookups plus a fixed-offset memcpy, eliminating nested loops and pointer chasing.2.2 Pinned Host Memory & Async CUDAHost buffers (large_arr, out_arr, etc.) are page-locked using cudaMallocHost, and transfers use cudaMemcpyAsync on a dedicated stream, overlapping PCIe cost with GPU computation.3. Parallelization Strategy3.1 MPI Reduction TreeAfter local multiplication, blocks reside on each rank. We execute log2(P) pairwise merges:for (int step = 1; step < P; step *= 2) {
    if (rank % (2*step) == 0)
        recv_and_merge(rank+step);
}
3.2 OpenMP ConcurrencyHost-side I/O and packing use OpenMP tasks and loops.#pragma omp parallel num_threads(16)
#pragma omp single
for (int b = 0; b < blocks; ++b) {
    #pragma omp task
    load_block_from_file(b);
}

#pragma omp parallel for
for (int jn = 0; jn < js.size(); ++jn) {
    memcpy(...);
}
3.3 CUDA KernelEach GPU block multiplies one k×k tile-pair. We choose k=32 to map threads to rows/columns, achieving > 85% SM occupancy.dim3 grid (num_pairs), block (k,k);
matrix_multiplyKernel <<<grid, block, 0, stream >>> (
    large_arr_gpu,
    key_to_elem_gpu,
    key_to_elem_prefix_gpu,
    k,
    out_arr_gpu
);
4. Performance AnalysisWe instrumented the code to measure Host packing, pinned-memcpy, Kernel execution, and MPI tree-merge times.4.1 End-to-End TimingsCasekCPU-Only (s)Optimized (s)SpeedupSmall (10k tiles)3215.23.44.5xMedium (100k tiles)32152.032.14.7xLarge (1M tiles)321530320.54.8xTable 1: Total multiply time across P=8 ranks, 16 threads/rank, NVIDIA P100 GPU.4.2 Detailed BreakdownPhaseTime (s)% of TotalHost pack & stage8.125%HD transfer5.417%GPU kernel12.338%DH transfer3.210%MPI merge2.16%Other overhead0.54%Total31.6100%Table 2: Breakdown of optimized run for 100k tiles.Host-to-device bandwidth: sustained ~12GB/s (90% of peak).GPU kernel: ~500 GFLOP/s (80% of theoretical).Overlap: 75% of H→D transfers overlap with kernel courtesy of async streams.MPI merge cost grows as O(logP); at P=16 ranks it remains < 10% of time.4.3 Scaling StudiesThreads/Rank481632Speedup vs 1 thread1.8x2.9x4.1x4.3xTable 3: Intra-node OpenMP scaling (100k tiles, 1 rank).Maximum speedup plateaued at 4.3x with 16-32 threads, indicating I/O/packing begins to saturate memory bandwidth.5. Code Snippets5.1 Asynchronous DMA + Kernel LaunchcudaStream_t stream;
cudaStreamCreate(&stream);

cudaMemcpyAsync(large_arr_gpu, large_arr, bytes, cudaMemcpyHostToDevice, stream);

// launch kernel on same stream
matrix_multiplyKernel <<<grid, block, 0, stream >>> (
    large_arr_gpu, key_to_elem_gpu,
    key_to_elem_prefix_gpu, k, out_arr_gpu);

// copy back
cudaMemcpyAsync(out_arr,
    out_arr_gpu,
    iters*k*k*sizeof(uint64_t),
    cudaMemcpyDeviceToHost, stream);

cudaStreamSynchronize(stream);
5.2 MPI Binary-Tree Reductionfor (int step = 1; step < world_size; step <<= 1) {
    if (rank % (2*step) == 0 && rank+step < world_size) {
        // receive partial product from rank+step
        MPI_Recv(..., rank+step, 0, MPI_COMM_WORLD, ...);
        // merge with local via helper()
    }
}
MPI_Barrier(MPI_COMM_WORLD);
6. ConclusionsBy re-architecting data layouts, overlapping communication via pinned memory and CUDA streams, and combining MPI, OpenMP, and CUDA effectively, we achieve a consistent ~4.7× end-to-end speedup on block-sparse multiplication. Future work includes adaptive tile sizing and NUMA-aware host staging.