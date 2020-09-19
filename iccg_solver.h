#pragma once


#include <cublas_v2.h>
#include <cusparse.h>

#include <cuda_runtime.h>

class ICCGsolver
{
	/* Incomplete-Cholesky Preconditioned Conjugate Gradient

	This class is used to solve sparse lienar systems: Ax=b
	where 'A' is a sparse symmetric definite positive matrix
	and 'b' is a vector.
	*/
private:
	cublasHandle_t cubHandle;
	cusparseHandle_t cusHandle;
	cusparseMatDescr_t descr_A;
	cusparseMatDescr_t descr_L;

	// host data
	int N = 0; // array size
	int nz = 0; // number of non-zero entry
	int max_iter;
	int k;
	double tolerance;
	double alpha;
	double beta;
	double rTr;
	double pTq;
	double rho;    //rho{k}
	double rho_t;  //rho{k-1}
	const double one = 1.0;  // constant
	const double zero = 0.0;   // constant

	// device data
	double *d_ic = nullptr; // 
	double *d_x = nullptr;
	double *d_y = nullptr;
	double *d_z = nullptr;
	double *d_r = nullptr;
	double *d_rt = nullptr;
	double *d_xt = nullptr;
	double *d_q = nullptr;
	double *d_p = nullptr;
	

	bool release_cusHandle = false;
	bool release_cubHandle = false;	

public:
	ICCGsolver(int max_iter = 1000, double tol = 1e-12,
		cublasHandle_t cub_handle = NULL,
		cusparseHandle_t cus_handle = NULL);

	~ICCGsolver();

	// N: matrix size NxN
	// nz: number of non-zero terms
	// d_A: a pointer to an array of non-zero terms of matrix A on device
	// d_rowIdx: a pointer to row index of matrix A on device
	// d_colIdx: a pointer to col index of matrix A on device
	// d_b: a pointer to vector b on device
	// d_guess: (optional) the initial guess of the answer
	bool solve(int N, int nz,
		double *d_A, int *d_rowIdx, int *d_colIdx,
		double *d_b, double *d_guess);

	// get the pointer of x on GPU
	double *x_ptr();
	double err();
	int iter_count();

private:
	void allocate_memory();
	void free_memory();
	void allocate_nonzero_memory();
	void allocate_array_memory();
	void free_array_memory();
	void free_nonzero_memory();
	void check_and_resize(int N, int nz);
};