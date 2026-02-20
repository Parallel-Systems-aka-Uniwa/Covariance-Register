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

# README

## Covariance Register

This repository implements a **CUDA-based program** for calculating basic statistics and generating the **covariance matrix** of a 2D N×N integer array. The project was developed for the **Parallel Systems** course at the **University of West Attica**.

---

## Table of Contents

| Section | Folder/File                           | Description                                                   |
| ------: | ------------------------------------- | ------------------------------------------------------------- |
|       1 | `assign/`                             | Assignment material for the Covariance Register CUDA workshop |
|     1.1 | `assign/_Par_Sys_Ask_2-2_2024-25.pdf` | Assignment description in English                             |
|     1.2 | `assign/_Παρ_Συσ_Ασκ_2-2_2024-25.pdf` | Assignment description in Greek                               |
|       2 | `docs/`                               | Documentation for covariance register computations            |
|     2.1 | `docs/Covariance-Register.pdf`        | English documentation                                         |
|     2.2 | `docs/Μητρώο-Συνδιακύμανσης.pdf`      | Greek documentation                                           |
|       3 | `src/`                                | Source code and input/output files                            |
|     3.1 | `src/cuda2.cu`                        | Main CUDA program for covariance register                     |
|     3.2 | `src/A/`                              | Input data files for exercise A                               |
|   3.2.1 | `src/A/A8.txt`                        | Input matrix A for 8 elements                                 |
|   3.2.2 | `src/A/A_cov8.txt`                    | Covariance matrix input for 8 elements                        |
|     3.3 | `src/A_cov/`                          | Intermediate covariance results                               |
|   3.3.1 | `src/A_cov/g.txt`                     | Covariance computation result                                 |
|     3.4 | `src/A_means/`                        | Mean values for A                                             |
|   3.4.1 | `src/A_means/A_means8.txt`            | Mean values for 8 elements                                    |
|   3.4.2 | `src/A_means/j.txt`                   | Additional mean data                                          |
|     3.5 | `src/A_submeans/`                     | Subtracted mean matrices for A                                |
|   3.5.1 | `src/A_submeans/A_submeans8.txt`      | Subtracted mean for 8 elements                                |
|   3.5.2 | `src/A_submeans/l.txt`                | Additional submean data                                       |
|     3.6 | `src/AT_submeans/`                    | Transposed subtracted mean matrices                           |
|   3.6.1 | `src/AT_submeans/AT_submeans8.txt`    | Transposed subtracted mean for 8 elements                     |
|   3.6.2 | `src/AT_submeans/l.txt`               | Additional transposed submean data                            |
|     3.7 | `src/Output/`                         | Output result files from CUDA execution                       |
|   3.7.1 | `src/Output/Output_no_args.txt`       | Output without arguments                                      |
|   3.7.2 | `src/Output/Output8.txt`              | Output for 8 elements                                         |
|   3.7.3 | `src/Output/Output512.txt`            | Output for 512 elements                                       |
|   3.7.4 | `src/Output/Output1024.txt`           | Output for 1024 elements                                      |
|   3.7.5 | `src/Output/Output10000.txt`          | Output for 10000 elements                                     |
|       4 | `README.md`                           | Project documentation                                         |
|       5 | `INSTALL.md`                          | Usage instructions                                            |

---

## 1. Overview

The program performs sequential operations on an input matrix to compute its covariance:

- **Column Means**: Computes the mean of each column.
- **Mean Subtraction & Transpose**: Subtracts the column mean from each element and creates a transposed difference matrix.
- **Covariance Calculation**: Generates the final covariance matrix using parallel computations.

---

## 2. Design & Implementation

**CUDA Kernels**:

- **`calcColMeans`**: Each thread processes a column to calculate its mean.
- **`subMeansT`**: Organized threads remove column means and transpose the matrix simultaneously.
- **`calcCov`**: Calculates the upper triangular covariance matrix while exploiting symmetry for performance.

**Technical Specifications**:

- **Synchronization**: Uses `__syncthreads()` to coordinate threads within a block.
- **Memory Management**: Employs shared memory for intermediate storage and faster access.
- **Optimization**: Grid and block configurations (`dimGrid`, `dimBlock`) maximize GPU resource usage.
