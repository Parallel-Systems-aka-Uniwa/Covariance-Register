/*
 *  === File: cuda2.cu ===
 *
 *  Full Name: Athanasiou Vasileios Evangelos
 *  Student ID: 19390005
 *  Degree Program: PADA
 *  
 *  Compilation: nvcc -o cuda2 cuda2.cu
 *  Execution: ./cuda2 A.txt A_means.txt A_submeans.txt AT_submeans.txt A_cov.txt
 * 
 */
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define N 10
#define nThreads 2
#define nBlocks (int)ceil((float)N/nThreads)

/*
 * === Function: calcColMeans ===
 * Parameters: 
 *   - d_A: Input array (Device) with integer elements.
 *   - d_Amean: Output array (Device) that stores the mean of each column.
 * Returns: Nothing.
 * 
 * Description:
 *   This function computes the mean of each column of the array d_A and stores
 *   the results in the array d_Amean. Each thread is responsible for one column.
 */
__global__ void calcColMeans(int *d_A, float *d_Amean) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Compute the global column index
    int row; // Row index

    if (col >= N) return; // Ensure we do not exceed the array bounds

    float sum = 0.0f;

    // Compute the sum of the column
    for (row = 0; row < N; ++row) 
        sum += d_A[row * N + col];

    // Store the result in the output array (compute the mean)
    d_Amean[col] = sum / N;
}

/*
 * === Function: subMeansT ===
 * Parameters: 
 *   - d_A: Input array (Device) with integer elements.
 *   - d_Amean: Input array (Device) containing the column means of d_A.
 *   - d_Asubmeans: Output array (Device) that stores the values of d_A after subtracting the mean.
 *   - d_ATsubmeans: Output transposed array (Device) that contains the transpose of d_Asubmeans.
 * Returns: Nothing.
 * 
 * Description:
 *   This function subtracts the mean of each column from the elements of d_A and stores
 *   the results in d_Asubmeans. In addition, it computes and stores the transpose of d_Asubmeans
 *   in d_ATsubmeans.
 */
__global__ void subMeansT(int *d_A, float *d_Amean, float *d_Asubmeans, float *d_ATsubmeans)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Compute the global row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Compute the global column index

    if (row < N && col < N) 
    {
        // Subtract the column mean from each element and store the result
        d_Asubmeans[row * N + col] = d_A[row * N + col] - d_Amean[col];
        // Compute and store the transpose of the array
        d_ATsubmeans[col * N + row] = d_Asubmeans[row * N + col];
    }
}

/*
 * === Function: calcCov ===
 * Parameters: 
 *   - d_Asubmeans: Input array (Device) containing the values of the original array with the column means subtracted.
 *   - d_Acov: Output array (Device) that will contain the computed covariance matrix.
 * Returns: Nothing.
 * 
 * Description:
 *   This function computes the covariance matrix for the data in d_Asubmeans.
 *   Only the elements of the upper triangular part of the covariance matrix are computed,
 *   since the matrix is symmetric.
 */
__global__ void calcCov(float *d_Asubmeans, float *d_Acov)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Compute row index for the covariance matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Compute column index for the covariance matrix
    float sum; // Accumulator
    int k; // Loop index

    if (row < N && col < N && row <= col) // Compute only for the upper triangular part
    { 
        sum = 0.0f;

        // Compute the inner product for row 'row' and column 'col'
        for (k = 0; k < N; ++k) 
            sum += d_Asubmeans[row * N + k] * d_Asubmeans[col * N + k];

        // Store the result in the covariance matrix
        d_Acov[row * N + col] = sum;
    }
}

/*
 * === Host Function: create2DArray ===
 * Parameters: 
 *      - Array: Pointer to a one-dimensional array (treated as 2D).
 * Returns: Nothing.
 * 
 * Description:
 *      Creates a random NxN array with values in the range [1, 100].
 *      Ensures that the maximum element of the array is either greater than or less than N times the 
 *      average of the array.
 */
void create2DArray(int *Array)
{
    int sum = 0;  // Sum of array elements
    int amax = 0; // Maximum value in the array
    int i, j, m;

    // Fill the array with random values and compute the sum and maximum value
    for (i = 0; i < N; ++i) 
    {
        for (j = 0; j < N; ++j) 
        {
            Array[i * N + j] = rand() % 100 + 1; // Random value in [1, 100]
            sum += Array[i * N + j]; // Add to total sum
            if (Array[i * N + j] > amax) 
            {
                amax = Array[i * N + j]; // Update maximum value
            }
        }
    }

    m = sum / (N * N); // Compute the average
    while (amax <= N * m) // Verify that the maximum value is greater than N * m
    {
        i = rand() % N; // Randomly select a row
        j = rand() % N; // Randomly select a column
        Array[i * N + j] += (N * m - amax + 1); // Increase the element to satisfy the condition
        amax = Array[i * N + j]; // Update maximum value
    }
}

/*
 * === Host Main Function ===
 * This host code uses the CUDA kernels above to compute:
 *  - The mean of each column of array A (calcColMeans)
 *  - The column-wise subtraction and transpose (subMeansT)
 *  - The covariance matrix (calcCov)
 */
int main(int argc, char *argv[])
{
    int *h_A;                                                        // [Host] Input array A
    float *h_Acov, *h_Amean, *h_Asubmeans, *h_ATsubmeans;            // [Host] Covariance matrix, column means, subtracted array, and transposed subtracted array
    int *d_A;                                                        // [Device] Input array A
    float *d_Acov, *d_Amean, *d_Asubmeans, *d_ATsubmeans;            // [Device] Covariance matrix, column means, subtracted array, and transposed subtracted array
    int n, threadsPerBlock, blocksPerGrid;                           // Array size, threads per block, blocks per grid
    int intBytes, floatBytes;                                        // Array sizes in bytes for integers and floats
    int max_threads, max_block_dimX, max_block_dimY, max_block_dimZ; // Maximum values for threads and block dimensions
    int max_grid_dimX, max_grid_dimY, max_grid_dimZ;                 // Maximum values for grid dimensions
    int i, j;                                                        // Loop indices for for-loops
    FILE *fpA, *fpAmean, *fpAsubmeans, *fpATsubmeans, *fpAcov;         // File pointers for storing arrays A, A_means, A_submeans, AT_submeans, and A_cov
    float elapsedTime1, elapsedTime2, elapsedTime3;                  // Kernel execution times
    cudaEvent_t start, stop;                                         // CUDA events for timing kernel execution
    cudaError_t err;                                                 // CUDA error code
    cudaDeviceProp prop;                                             // CUDA device properties

    // Check command-line parameter count
    if (argc != 6)
    {
        printf("Usage: %s A.txt A_means.txt A_submeans.txt AT_submeans.txt A_cov.txt\n", argv[0]);
        exit(1);
    }

    // Initialize parameters
    n = N;
    threadsPerBlock = nThreads;
    blocksPerGrid = nBlocks;

    // Get CUDA device properties
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaGetDeviceProperties() failed.\n"); exit(1); } 

    // Assign maximum values from device properties
    max_threads = prop.maxThreadsPerBlock;
    max_block_dimX = prop.maxThreadsDim[0];
    max_block_dimY = prop.maxThreadsDim[1];
    max_block_dimZ = prop.maxThreadsDim[2];
    max_grid_dimX = prop.maxGridSize[0];
    max_grid_dimY = prop.maxGridSize[1];
    max_grid_dimZ = prop.maxGridSize[2];

    // Print device properties
    printf("--------------- Device Properties ---------------\n");
    printf("Device name           : %s\n", prop.name);
    printf("Max threads per block : %d\n", max_threads);
    printf("Max block dimensions  : %d x %d x %d\n", max_block_dimX, max_block_dimY, max_block_dimZ);
    printf("Max grid dimensions   : %d x %d x %d\n", max_grid_dimX, max_grid_dimY, max_grid_dimZ);
    printf("-------------------------------------------------\n");

    // Validate parameter values
    if (n < 1)
    { printf("Error --> Matrix size must be at least 1\n"); exit(1); }
    if (threadsPerBlock < 1)
    { printf("Error --> Threads per block (block size) must be at least 1\n"); exit(1); }
    if (blocksPerGrid < 1)
    { printf("Error --> Blocks per grid (grid size) must be at least 1\n"); exit(1); }
    if (threadsPerBlock > max_threads)
    { printf("Error --> Threads per block (block size) exceed maximum allowed for %s\n", prop.name); exit(1); }
    if (blocksPerGrid > max_grid_dimX)
    { printf("Error --> Blocks per grid (grid size) exceed maximum allowed for %s\n", prop.name); exit(1); }

    // Open files to store arrays
    fpA = fopen(argv[1], "w");
    if (fpA == NULL) { printf("Cannot open file %s\n", argv[1]); exit(1); }
    fpAmean = fopen(argv[2], "w");
    if (fpAmean == NULL) { printf("Cannot open file %s\n", argv[2]); exit(1); }
    fpAsubmeans = fopen(argv[3], "w");
    if (fpAsubmeans == NULL) { printf("Cannot open file %s\n", argv[3]); exit(1); }
    fpATsubmeans = fopen(argv[4], "w");
    if (fpATsubmeans == NULL) { printf("Cannot open file %s\n", argv[4]); exit(1); }
    fpAcov = fopen(argv[5], "w");
    if (fpAcov == NULL) { printf("Cannot open file %s\n", argv[5]); exit(1); }
  
    // Print input parameters
    printf("--------------- Input Parameters ---------------\n");
    printf("Matrix size        : %d x %d\n", n, n);
    printf("Blocks per Grid    : %d\n", blocksPerGrid);
    printf("Threads per Block  : %d\n", threadsPerBlock);
    printf("------------------------------------------------\n");

    // Calculate array sizes in bytes
    intBytes = n * n * sizeof(int);
    floatBytes = n * n * sizeof(float);

    // Allocate host memory for arrays
    h_A = (int *) malloc(intBytes);
    if (h_A == NULL) { printf("Error --> Memory allocation failed for A.\n"); exit(1); }
    h_Amean = (float *) malloc(n * sizeof(float));
    if (h_Amean == NULL) { printf("Error --> Memory allocation failed for A_mean.\n"); exit(1); }
    h_Asubmeans = (float *) malloc(floatBytes);
    if (h_Asubmeans == NULL) { printf("Error --> Memory allocation failed for A_submeans.\n"); exit(1); }
    h_ATsubmeans = (float *) malloc(floatBytes);
    if (h_ATsubmeans == NULL) { printf("Error --> Memory allocation failed for AT_submeans.\n"); exit(1); }
    h_Acov = (float *) malloc(floatBytes);
    if (h_Acov == NULL) { printf("Error --> Memory allocation failed for A_cov.\n"); exit(1); }

    srand(time(NULL));

    // Create random array A and initialize other arrays to zero
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            h_A[i * n + j] = rand() % 199 - 99;                           // Values in the range [-99, 99]
            h_A[i * n + j] = h_A[i * n + j] >= 0 ? h_A[i * n + j] + 10 : h_A[i * n + j] - 10;  // Randomly adjust the sign
            h_Asubmeans[i * n + j] = 0.0;
            h_ATsubmeans[i * n + j] = 0.0;
            h_Acov[i * n + j] = 0.0;
        }
        h_Amean[i] = 0.0;
    }
    printf("The array A has been stored in file %s\n", argv[1]);

// ============== Start of Parallel Computation ============== 

    // Create CUDA events for timing
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventCreate(&start) failed.\n"); exit(1); }
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventCreate(&stop) failed.\n"); exit(1); }

    // Allocate device memory
    err = cudaMalloc((void **) &d_A, intBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_A, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_Amean, n * sizeof(float));
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_Amean, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_Asubmeans, floatBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_Asubmeans, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_ATsubmeans, floatBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_ATsubmeans, bytes) failed."); exit(1); }
    err = cudaMalloc((void **) &d_Acov, floatBytes);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMalloc((void **) &d_Acov, bytes) failed."); exit(1); }

    err = cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice) failed."); exit(1); }

    // Create 2D grid with 2D blocks
    dim3 dimBlock(nThreads, nThreads);
    dim3 dimGrid(nBlocks, nBlocks);

/* 
 * === Execute Kernel: calcColMeans<<<dimGrid, dimBlock>>> ===
 * Purpose:
 *   - Computes the mean of each column of d_A and stores the results in d_Amean.
 *
 * Grid and Block Configuration:
 *   - dimGrid  : Represents the number of blocks in the grid (X: nBlocks, Y: 1, Z: 1).
 *   - dimBlock : Represents the number of threads per block (X: nThreads, Y: 1, Z: 1).
 * 
 * Memory Transfers:
 *   - cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice) transfers the input array from host to device.
 *   - cudaMemcpy(h_Amean, d_Amean, n * sizeof(float), cudaMemcpyDeviceToHost) retrieves the computed column means from device to host.
 *
 * Performance Measurement:
 *   - Uses CUDA events to measure kernel execution time.
 *
 * Error Handling:
 *   - Each CUDA call is followed by an error check (if not cudaSuccess).
 */
    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start, 0) failed."); exit(1); }

    calcColMeans<<<dimGrid, nThreads>>>(d_A, d_Amean);

    err = cudaEventRecord(stop, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(stop, 0) failed."); exit(1); }

    err = cudaMemcpy(h_Amean, d_Amean, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(h_Amean, d_Amean, bytes, cudaMemcpyDeviceToHost) failed."); exit(1); }

    printf("The array A_means has been stored in file %s\n", argv[2]);

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventSynchronize(stop) failed."); exit(1); }
    err = cudaEventElapsedTime(&elapsedTime1, start, stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventElapsedTime(&elapsedTime1, start, stop) failed."); exit(1); }
    
    printf("Time for the kernel calcColMeans<<<>>>(): %f ms\n", elapsedTime1);

/* 
 * === Execute Kernel: subMeansT<<<dimGrid, dimBlock>>> ===
 * Purpose:
 *   - Computes the difference between each element of d_A and the mean of its column,
 *     and stores the results in d_Asubmeans.
 *   - Also creates the transpose of d_Asubmeans in d_ATsubmeans.
 *
 * Grid and Block Configuration:
 *   - dimGrid  : Represents the number of blocks in the grid (X: nBlocks, Y: nBlocks, Z: 1).
 *   - dimBlock : Represents the number of threads per block (X: nThreads, Y: nThreads, Z: 1).
 * 
 * Memory Transfers:
 *   - cudaMemcpy(d_A, h_A, intBytes, cudaMemcpyHostToDevice) transfers the input array from host to device.
 *   - cudaMemcpy(h_Asubmeans, d_Asubmeans, floatBytes, cudaMemcpyDeviceToHost) retrieves d_Asubmeans from device to host.
 *   - cudaMemcpy(h_ATsubmeans, d_ATsubmeans, floatBytes, cudaMemcpyDeviceToHost) retrieves d_ATsubmeans from device to host.
 *
 * Performance Measurement:
 *   - Uses CUDA events to measure kernel execution time.
 *
 * Error Handling:
 *   - Each CUDA call is followed by an error check.
 */
    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start, 0) failed."); exit(1); }

    subMeansT<<<dimGrid, dimBlock>>>(d_A, d_Amean, d_Asubmeans, d_ATsubmeans);

    err = cudaEventRecord(stop, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(stop, 0) failed."); exit(1); }

    err = cudaMemcpy(h_Asubmeans, d_Asubmeans, floatBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(h_Asubmeans, d_Asubmeans, bytes, cudaMemcpyDeviceToHost) failed."); exit(1); }
    err = cudaMemcpy(h_ATsubmeans, d_ATsubmeans, floatBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(h_ATsubmeans, d_ATsubmeans, bytes, cudaMemcpyDeviceToHost) failed."); exit(1); }

    printf("The array A_submeans has been stored in file %s\n", argv[3]);
    printf("The array AT_submeans has been stored in file %s\n", argv[4]);

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventSynchronize(stop) failed."); exit(1); }
    err = cudaEventElapsedTime(&elapsedTime2, start, stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventElapsedTime(&elapsedTime2, start, stop) failed."); exit(1); }
    
    printf("Time for the kernel subMeansT<<<>>>(): %f ms\n", elapsedTime2);

/* 
 * === Execute Kernel: calcCov<<<dimGrid, dimBlock>>> ===
 * Purpose:
 *   - Computes the covariance matrix d_Acov from the array d_Asubmeans,
 *     which contains the values of the original array with the column means subtracted.
 *
 * Grid and Block Configuration:
 *   - dimGrid  : Represents the number of blocks in the grid (X: nBlocks, Y: nBlocks, Z: 1).
 *   - dimBlock : Represents the number of threads per block (X: nThreads, Y: nThreads, Z: 1).
 * 
 * Memory Transfers:
 *   - cudaMemcpy(h_Acov, d_Acov, floatBytes, cudaMemcpyDeviceToHost) retrieves the computed covariance matrix from device to host.
 *
 * Performance Measurement:
 *   - Uses CUDA events to measure kernel execution time.
 *
 * Error Handling:
 *   - Each CUDA call is followed by an error check.
 */
    err = cudaEventRecord(start, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(start, 0) failed."); exit(1); }

    calcCov<<<dimGrid, dimBlock>>>(d_Asubmeans, d_Acov);

    err = cudaEventRecord(stop, 0);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventRecord(stop, 0) failed."); exit(1); }

    err = cudaMemcpy(h_Acov, d_Acov, floatBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaMemcpy(h_Acov, d_Acov, bytes, cudaMemcpyDeviceToHost) failed."); exit(1); }

    printf("The array A_cov has been stored in file %s\n", argv[5]);

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventSynchronize(stop) failed."); exit(1); }
    err = cudaEventElapsedTime(&elapsedTime3, start, stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventElapsedTime(&elapsedTime3, start, stop) failed."); exit(1); }
    
    printf("Time for the kernel calcCov<<<>>>(): %f ms\n", elapsedTime3);

// ============== End of Parallel Computation ==============

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            fprintf(fpA, "%4d ", h_A[i * n + j]);
            fprintf(fpAsubmeans, "%4.4f ", h_Asubmeans[i * n + j]);
            fprintf(fpATsubmeans, "%4.4f ", h_ATsubmeans[i * n + j]);
            fprintf(fpAcov, "%4.4f ", h_Acov[i * n + j]);
        }
        fprintf(fpAmean, "%4.4f\n", h_Amean[i]);
        fprintf(fpA, "\n");
        fprintf(fpAsubmeans, "\n");
        fprintf(fpATsubmeans, "\n");
        fprintf(fpAcov, "\n");
    }

    fclose(fpA);
    fclose(fpAmean);
    fclose(fpAcov);
    fclose(fpAsubmeans);
    fclose(fpATsubmeans);

    err = cudaEventDestroy(start);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventDestroy(start) failed."); exit(1); }
    err = cudaEventDestroy(stop);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaEventDestroy(stop) failed."); exit(1); }

    free(h_A);
    free(h_Acov);
    free(h_Amean);
    free(h_Asubmeans);
    free(h_ATsubmeans);
    
    err = cudaFree(d_A);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_A) failed."); exit(1); }
    err = cudaFree(d_Acov);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_Acov) failed."); exit(1); }
    err = cudaFree(d_Amean);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_Amean) failed."); exit(1); }
    err = cudaFree(d_Asubmeans);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_Asubmeans) failed."); exit(1); }
    err = cudaFree(d_ATsubmeans);
    if (err != cudaSuccess) { printf("CUDA Error --> cudaFree(d_ATsubmeans) failed."); exit(1); }

    return 0;
}
