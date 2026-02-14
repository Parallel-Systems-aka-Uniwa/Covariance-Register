<p align="center">
  <img src="https://www.especial.gr/wp-content/uploads/2019/03/panepisthmio-dut-attikhs.png" alt="UNIWA" width="150"/>
</p>

<p align="center">
  <strong>UNIVERSITY OF WEST ATTICA</strong><br>
  SCHOOL OF ENGINEERING<br>
  DEPARTMENT OF COMPUTER ENGINEERING AND INFORMATICS
</p>

---

<p align="center">
  <strong>Parallel Systems</strong>
</p>

<h1 align="center">
  Covariance Register
</h1>

<p align="center">
  <strong>Vasileios Evangelos Athanasiou</strong><br>
  Student ID: 19390005
</p>

<p align="center">
  <a href="https://github.com/Ath21" target="_blank">GitHub</a> ·
  <a href="https://www.linkedin.com/in/vasilis-athanasiou-7036b53a4/" target="_blank">LinkedIn</a>
</p>

<hr/>

<p align="center">
  <strong>Supervision</strong>
</p>

<p align="center">
  Supervisor: Vasileios Mamalis, Professor
</p>
<p align="center">
  <a href="https://ice.uniwa.gr/en/emd_person/vassilios-mamalis/" target="_blank">UNIWA Profile</a>
</p>


<p align="center">
  Co-supervisor: Michalis Iordanakis, Special Technical Laboratory Staff
</p>

<p align="center">
  <a href="https://scholar.google.com/citations?user=LiVuwVEAAAAJ&hl=en" target="_blank">UNIWA Profile</a>
</p>

</hr>

<p align="center">
  Athens, February 2025
</p>


---

# Covariance Register

This repository implements a **CUDA-based program** for calculating basic statistics and generating the **covariance matrix** of a 2D N×N integer array. The project was developed for the **Parallel Systems** course at the **University of West Attica**.

---

## Table of Contents

| Section | Folder/File | Description |
|------:|-------------|-------------|
| 1 | `assign/` | Assignment material for the Covariance Register CUDA workshop |
| 1.1 | `assign/_Par_Sys_Ask_2-2_2024-25.pdf` | Assignment description in English |
| 1.2 | `assign/_Παρ_Συσ_Ασκ_2-2_2024-25.pdf` | Assignment description in Greek |
| 2 | `docs/` | Documentation for covariance register computations |
| 2.1 | `docs/Covariance-Register.pdf` | English documentation |
| 2.2 | `docs/Μητρώο-Συνδιακύμανσης.pdf` | Greek documentation |
| 3 | `src/` | Source code and input/output files |
| 3.1 | `src/cuda2.cu` | Main CUDA program for covariance register |
| 3.2 | `src/A/` | Input data files for exercise A |
| 3.2.1 | `src/A/A8.txt` | Input matrix A for 8 elements |
| 3.2.2 | `src/A/A_cov8.txt` | Covariance matrix input for 8 elements |
| 3.3 | `src/A_cov/` | Intermediate covariance results |
| 3.3.1 | `src/A_cov/g.txt` | Covariance computation result |
| 3.4 | `src/A_means/` | Mean values for A |
| 3.4.1 | `src/A_means/A_means8.txt` | Mean values for 8 elements |
| 3.4.2 | `src/A_means/j.txt` | Additional mean data |
| 3.5 | `src/A_submeans/` | Subtracted mean matrices for A |
| 3.5.1 | `src/A_submeans/A_submeans8.txt` | Subtracted mean for 8 elements |
| 3.5.2 | `src/A_submeans/l.txt` | Additional submean data |
| 3.6 | `src/AT_submeans/` | Transposed subtracted mean matrices |
| 3.6.1 | `src/AT_submeans/AT_submeans8.txt` | Transposed subtracted mean for 8 elements |
| 3.6.2 | `src/AT_submeans/l.txt` | Additional transposed submean data |
| 3.7 | `src/Output/` | Output result files from CUDA execution |
| 3.7.1 | `src/Output/Output_no_args.txt` | Output without arguments |
| 3.7.2 | `src/Output/Output8.txt` | Output for 8 elements |
| 3.7.3 | `src/Output/Output512.txt` | Output for 512 elements |
| 3.7.4 | `src/Output/Output1024.txt` | Output for 1024 elements |
| 3.7.5 | `src/Output/Output10000.txt` | Output for 10000 elements |
| 4 | `README.md` | Repository overview and usage instructions |
---

## Overview

The program performs sequential operations on an input matrix to compute its covariance:

- **Column Means**: Computes the mean of each column.
- **Mean Subtraction & Transpose**: Subtracts the column mean from each element and creates a transposed difference matrix.
- **Covariance Calculation**: Generates the final covariance matrix using parallel computations.

---

## Design & Implementation

**CUDA Kernels**:

- **`calcColMeans`**: Each thread processes a column to calculate its mean.
- **`subMeansT`**: Organized threads remove column means and transpose the matrix simultaneously.
- **`calcCov`**: Calculates the upper triangular covariance matrix while exploiting symmetry for performance.

**Technical Specifications**:

- **Synchronization**: Uses `__syncthreads()` to coordinate threads within a block.
- **Memory Management**: Employs shared memory for intermediate storage and faster access.
- **Optimization**: Grid and block configurations (`dimGrid`, `dimBlock`) maximize GPU resource usage.

---

# Installation & Setup Guide

This repository implements **parallel computations on 2D integer arrays** using **CUDA**, developed as part of the **Parallel Systems course** at the **University of West Attica**. The project demonstrates GPU-accelerated matrix operations using **CUDA threads, shared memory, and atomic operations**.

---

## Prerequisites

### Required Software
- **NVIDIA CUDA Toolkit** (≥ 11.0 recommended)  
  Download: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- **NVIDIA GPU** with compute capability ≥ 3.0 (tested on NVIDIA TITAN RTX)
- **GCC compiler** (Linux/macOS) or compatible compiler on Windows
- **Make / Terminal** for compilation and execution

### Optional Software
- Text editor or IDE (VSCode, CLion, Nsight)  
- Spreadsheet viewer for performance analysis

---

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Parallel-Systems-aka-Uniwa/Covariance-Register.git
```

Or download the ZIP archive and extract it.

### 2. Navigate to Project Directory
```bash
cd Covariance-Register
```

Folder structure:
```bash
assign/
docs/
src/
README.md
```

`src/` contains CUDA source code (cuda2.cu) and input/output directories

`docs/` contains theory, exercises, and performance analysis

---

### 3. Compile the CUDA Program
Make sure you have CUDA Toolkit installed and nvcc is available in your PATH.
```bash
nvcc -o cuda2 src/cuda2.cu
```
This compiles the main program cuda2.cu into the executable cuda2.

Compilation options:
- `-O2` or `-O3` for optimization
- `-G` for debugging (optional)
- `-arch=sm_60` (or higher) to specify GPU architecture if needed:
```bash
nvcc -arch=sm_60 -O3 -o cuda2 src/cuda2.cu
```

Verify compilation by listing the executable:
```bash
ls -l cuda2
```

### 4. Prepare Input Files
The repository provides sample input matrices in `src/A/`. Examples:

| File         | Description                         |
|-------------|-------------------------------------|
| A8.txt       | Input matrix with 8 elements        |
| A_cov8.txt   | Covariance matrix reference for 8 elements |

You can also create your own N×N integer matrix as a text file with elements separated by spaces or newlines.

### 5. Run the Program
### Basic Execution
```bash
./cuda2 src/A/A8.txt src/A_means/A_means8.txt src/A_submeans/A_submeans8.txt src/AT_submeans/AT_submeans8.txt src/A_cov/A_cov8.txt
```

Parameters:
- Input matrix (A.txt)
- Output column means (A_means.txt)
- Output subtracted means matrix (A_submeans.txt)
- Output transposed subtracted means (AT_submeans.txt)
- Output covariance matrix (A_cov.txt)

> Adjust filenames for different matrix sizes (e.g., A512.txt, A1024.txt, etc.)

### 6. Verify Output
Open the resulting `.txt` files to verify:

| Output File     | Purpose                              |
|----------------|--------------------------------------|
| A_means.txt     | Column-wise mean values              |
| A_submeans.txt  | Each element minus column mean       |
| AT_submeans.txt | Transpose of subtracted mean matrix  |
| A_cov.txt       | Covariance matrix of input           |

- For small matrices (N=8 or N=16), manually verify calculations.
- For large matrices (N≥1024), check sum of diagonal elements, variances, or use Python/Excel to verify correctness.

---

## Performance Results
Experiments were conducted on an NVIDIA TITAN RTX GPU. Execution times for different matrix sizes:

| Matrix Dimension | calcColMeans (ms) | subMeansT (ms) | calcCov (ms) |
|-----------------|-----------------|----------------|---------------|
| 8 × 8           | 0.253824        | 0.019936       | 0.021216      |
| 512 × 512       | 0.124000        | 0.018880       | 0.885408      |
| 1024 × 1024     | 0.159168        | 0.071616       | 11.949280     |
| 10000 × 10000   | 1.065632        | 0.009952       | 0.124096      |

Observations:
- **Scalability**: `calcColMeans` scales linearly with matrix size.
- **Startup Overhead**: Small matrices show higher relative times due to kernel startup latency.
- **Resource Efficiency**: `subMeansT` kernel performs better on larger matrices due to shared memory usage and GPU saturation.