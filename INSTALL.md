<p align="center">
  <img src="https://www.especial.gr/wp-content/uploads/2019/03/panepisthmio-dut-attikhs.png" alt="UNIWA" width="150"/>
</p>

<p align="center">
  <strong>UNIVERSITY OF WEST ATTICA</strong><br>
  SCHOOL OF ENGINEERING<br>
  DEPARTMENT OF COMPUTER ENGINEERING AND INFORMATICS
</p>

<p align="center">
  <a href="https://www.uniwa.gr" target="_blank">University of West Attica</a> ·
  <a href="https://ice.uniwa.gr" target="_blank">Department of Computer Engineering and Informatics</a>
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
  Co-supervisor: Michalis Iordanakis, Academic Scholar
</p>

<p align="center">
  <a href="https://ice.uniwa.gr/academic_sc_ho/" target="_blank">UNIWA Profile</a> ·
  <a href="https://scholar.google.com/citations?user=LiVuwVEAAAAJ&hl=en" target="_blank">Scholar</a>
</p>

</hr>

---

<p align="center">
  Athens, February 2025
</p>

---

<p align="center">
  <img src="https://ai-ml-analytics.com/wp-content/uploads/2020/06/collage-scaled.jpg" width="250"/>
</p>

---

# INSTALL

## Covariance Register

This repository implements **parallel computations on 2D integer arrays** using **CUDA**, developed as part of the **Parallel Systems course** at the **University of West Attica**. The project demonstrates GPU-accelerated matrix operations using **CUDA threads, shared memory, and atomic operations**.

---

## 1. Prerequisites

### 1.1 Required Software

- **NVIDIA CUDA Toolkit** (≥ 11.0 recommended)  
  Download: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- **NVIDIA GPU** with compute capability ≥ 3.0 (tested on NVIDIA TITAN RTX)
- **GCC compiler** (Linux/macOS) or compatible compiler on Windows
- **Make / Terminal** for compilation and execution

### 1.2 Optional Software

- Text editor or IDE (VSCode, CLion, Nsight)
- Spreadsheet viewer for performance analysis

---

## 2. Installation Steps

### 2.1 Clone the Repository

```bash
git clone https://github.com/Parallel-Systems-aka-Uniwa/Covariance-Register.git
```

Or download the ZIP archive and extract it.

### 2.2 Navigate to Project Directory

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

## 3. Compile the CUDA Program

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

---

## 4. Prepare Input Files

The repository provides sample input matrices in `src/A/`. Examples:

| File       | Description                                |
| ---------- | ------------------------------------------ |
| A8.txt     | Input matrix with 8 elements               |
| A_cov8.txt | Covariance matrix reference for 8 elements |

You can also create your own N×N integer matrix as a text file with elements separated by spaces or newlines.

---

## 5. Run the Program

### 5.1 Basic Execution

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

### 5.2 Verify Output

Open the resulting `.txt` files to verify:

| Output File     | Purpose                             |
| --------------- | ----------------------------------- |
| A_means.txt     | Column-wise mean values             |
| A_submeans.txt  | Each element minus column mean      |
| AT_submeans.txt | Transpose of subtracted mean matrix |
| A_cov.txt       | Covariance matrix of input          |

- For small matrices (N=8 or N=16), manually verify calculations.
- For large matrices (N≥1024), check sum of diagonal elements, variances, or use Python/Excel to verify correctness.

---

## 6. Performance Results

Experiments were conducted on an NVIDIA TITAN RTX GPU. Execution times for different matrix sizes:

| Matrix Dimension | calcColMeans (ms) | subMeansT (ms) | calcCov (ms) |
| ---------------- | ----------------- | -------------- | ------------ |
| 8 × 8            | 0.253824          | 0.019936       | 0.021216     |
| 512 × 512        | 0.124000          | 0.018880       | 0.885408     |
| 1024 × 1024      | 0.159168          | 0.071616       | 11.949280    |
| 10000 × 10000    | 1.065632          | 0.009952       | 0.124096     |

Observations:

- **Scalability**: `calcColMeans` scales linearly with matrix size.
- **Startup Overhead**: Small matrices show higher relative times due to kernel startup latency.
- **Resource Efficiency**: `subMeansT` kernel performs better on larger matrices due to shared memory usage and GPU saturation.
