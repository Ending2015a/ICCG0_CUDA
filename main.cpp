#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include "iccg_solver.h"
#include "error_helper.h"

void read(std::string filePath,
		  int *pN, int *pnz,
		  double **cooVal,
		  int **cooRowIdx, int **cooColIdx,
		  double **b)
{
	std::ifstream in(filePath, std::ios::binary);

	in.read((char*)pN, sizeof(int));
	in.read((char*)pnz, sizeof(int));

	*cooVal = new double[*pnz]{};
	*cooRowIdx = new int[*pnz]{};
	*cooColIdx = new int[*pnz]{};
	*b = new double[*pN]{};

	for (int i = 0; i < *pnz; ++i)
	{
		in.read((char*)&(*cooRowIdx)[i], sizeof(int));
		in.read((char*)&(*cooColIdx)[i], sizeof(int));
		in.read((char*)&(*cooVal)[i], sizeof(double));
	}

	in.read((char*)(*b), sizeof(double)*(*pN));
}

void readAnswer(std::string filePath,
				int *pN, double **x)
{
	std::ifstream in(filePath, std::ios::binary);

	in.read((char*)pN, sizeof(int));

	*x = new double[*pN]{};

	in.read((char*)(*x), sizeof(double)*(*pN));
}


int main(int argc, char **argv)
{
	std::string inputPath = "testcase/full/size1M/case_1M.in";
	std::string answerPath = "testcase/full/size1M/case_1M.out";

	int N;
	int nz;
	double *A;
	int *rowIdxA;
	int *colIdxA;
	double *b;
	read(inputPath, &N, &nz, &A, &rowIdxA, &colIdxA, &b);

	double *ans_x;
	readAnswer(answerPath, &N, &ans_x);

	std::cout << "N = " << N << std::endl;
	std::cout << "nz = " << nz << std::endl;

	// Create handles
	cublasHandle_t cubHandle;
	cusparseHandle_t cusHandle;

	error_check(cublasCreate(&cubHandle));
	error_check(cusparseCreate(&cusHandle));

	// Allocate GPU memory & copy matrix/vector to device
	double *d_A;
	int *d_rowIdxA; // COO
	int *d_rowPtrA; // CSR
	int *d_colIdxA;
	double *d_b;

	error_check(cudaMalloc(&d_A, nz * sizeof(double)));
	error_check(cudaMalloc(&d_rowIdxA, nz * sizeof(int)));
	error_check(cudaMalloc(&d_rowPtrA, (N + 1) * sizeof(int)));
	error_check(cudaMalloc(&d_colIdxA, nz * sizeof(int)));
	error_check(cudaMalloc(&d_b, N * sizeof(double)));

	error_check(cudaMemcpy(d_A, A, nz * sizeof(double), cudaMemcpyHostToDevice));
	error_check(cudaMemcpy(d_rowIdxA, rowIdxA, nz * sizeof(int), cudaMemcpyHostToDevice));
	error_check(cudaMemcpy(d_colIdxA, colIdxA, nz * sizeof(int), cudaMemcpyHostToDevice));
	error_check(cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice));

	// Convert matrix A from COO format to CSR format
	error_check(cusparseXcoo2csr(cusHandle, d_rowIdxA, nz, N,
						d_rowPtrA, CUSPARSE_INDEX_BASE_ZERO));

	ICCGsolver solver(1000, 1e-12, cubHandle, cusHandle);

	std::cout << "Solving..." << std::endl;
	bool res = solver.solve(N, nz, d_A, d_rowPtrA, d_colIdxA, d_b, NULL);

	if (res)
		std::cout << "Converged!" << std::endl;
	else
		std::cout << "Failed to converge" << std::endl;

	double *x = new double[N] {};
	error_check(cudaMemcpy(x, solver.x_ptr(), N * sizeof(double), cudaMemcpyDeviceToHost));

	double tol = 0;
	for (int i = 0; i < N; ++i)
	{
		tol += fabs(x[i] - ans_x[i]);
	}

	// print message
	std::cout << "Solved in " << solver.iter_count() << " iterations, final norm(r) = "
		<< std::scientific << solver.err() << std::endl;

	std::cout << "Total error (compared with ans_x): " << tol << std::endl;

	// Free Host memory
	delete[] A;
	delete[] rowIdxA;
	delete[] colIdxA;
	delete[] b;
	delete[] ans_x;
	delete[] x;

	// Free Device memory
	cudaFree(d_A);
	cudaFree(d_rowIdxA);
	cudaFree(d_rowPtrA);
	cudaFree(d_colIdxA);
	cudaFree(d_b);

	// Free handles
	cublasDestroy(cubHandle);
	cusparseDestroy(cusHandle);

	return 0;
}