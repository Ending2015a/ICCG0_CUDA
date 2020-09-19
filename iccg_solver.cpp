
#define ERROR_HELPER_DONT_IMPLEMENT

#include "iccg_solver.h"
#include "error_helper.h"

#include <iostream>

ICCGsolver::ICCGsolver(int max_iter, double tol,
	cublasHandle_t cub_handle, cusparseHandle_t cus_handle) 
	: max_iter(max_iter), tolerance(tol), 
	  cubHandle(cub_handle), cusHandle(cus_handle)
{
	// create cuBLAS handle
	if (cubHandle == NULL)
	{
		error_check(cublasCreate(&cubHandle));
		release_cubHandle = true;
	}
		

	// create cuSPARSE handle
	if (cusHandle == NULL)
	{
		error_check(cusparseCreate(&cusHandle));
		release_cusHandle = true;
	}
	
	// create descriptor for matrix A
	error_check(cusparseCreateMatDescr(&descr_A));

	// initialize properties of matrix A
	error_check(cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL));
	error_check(cusparseSetMatFillMode(descr_A, CUSPARSE_FILL_MODE_LOWER));
	error_check(cusparseSetMatDiagType(descr_A, CUSPARSE_DIAG_TYPE_NON_UNIT));
	error_check(cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO));

	// create descriptor for matrix L
	error_check(cusparseCreateMatDescr(&descr_L));

	// initialize properties of matrix L
	error_check(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
	error_check(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER));
	error_check(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO));
	error_check(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT));
}

ICCGsolver::~ICCGsolver()
{
	// free data
	free_memory();

	// release descriptor
	cusparseDestroyMatDescr(descr_A);
	cusparseDestroyMatDescr(descr_L);

	// release handles
	if (release_cubHandle)
	{
		cublasDestroy(cubHandle);
	}

	if (release_cusHandle)
	{
		cusparseDestroy(cusHandle);
	}
}

bool ICCGsolver::solve(int N, int nz,
	double *d_A, int *d_rowIdx, int *d_colIdx,
	double *d_b, double *d_guess)
{
	check_and_resize(N, nz);

	/* --- Create cuSPARSE generic API objects --- */

	cusparseSpMatDescr_t smat_A;
	error_check(cusparseCreateCsr(&smat_A, N, N, nz, d_rowIdx, d_colIdx, d_A, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

	cusparseDnVecDescr_t dvec_p;
	error_check(cusparseCreateDnVec(&dvec_p, N, d_p, CUDA_R_64F));

	cusparseDnVecDescr_t dvec_q;
	error_check(cusparseCreateDnVec(&dvec_q, N, d_q, CUDA_R_64F));

	cusparseDnVecDescr_t dvec_x;
	error_check(cusparseCreateDnVec(&dvec_x, N, d_x, CUDA_R_64F));


	/* --- Perform incomplete-cholesky factorization --- */

	csric02Info_t icinfo_A;
	size_t buf_size = 0;
	size_t u_temp_buf_size = 0;
	size_t u_temp_buf_size2 = 0;
	int i_temp_buf_size = 0;
	void *d_buf = NULL;

	// Create info object for incomplete-cholesky factorization
	error_check(cusparseCreateCsric02Info(&icinfo_A));
	// Compute buffer size in computing ic factorization
	error_check(cusparseDcsric02_bufferSize(cusHandle, N, nz, 
		descr_A, d_A, d_rowIdx, d_colIdx, icinfo_A, &i_temp_buf_size));
	buf_size = i_temp_buf_size;

	// Create buffer
	error_check(cudaMalloc(&d_buf, buf_size));
	// Copy A
	error_check(cudaMemcpy(d_ic, d_A, nz * sizeof(double), cudaMemcpyDeviceToDevice));

	// Perform incomplete-choleskey factorization: analysis phase
	error_check(cusparseDcsric02_analysis(cusHandle, N, nz,
		descr_A, d_ic, d_rowIdx, d_colIdx, icinfo_A, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf));

	// Perform incomplete-choleskey factorization: solve phase
	error_check(cusparseDcsric02(cusHandle, N, nz,
		descr_A, d_ic, d_rowIdx, d_colIdx, icinfo_A, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf));

	// Create info object for factorized matrix L, LT
	csrsv2Info_t info_L, info_U;
	error_check(cusparseCreateCsrsv2Info(&info_L));
	error_check(cusparseCreateCsrsv2Info(&info_U));


	/* --- Compute buffer size for conjugate gradient comuptation --- */


	// Compute buffer size in solving linear system
	error_check(cusparseDcsrsv2_bufferSize(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
		N, nz, descr_L, d_ic, d_rowIdx, d_colIdx, info_L, &i_temp_buf_size));

	u_temp_buf_size = i_temp_buf_size;

	error_check(cusparseDcsrsv2_bufferSize(cusHandle, CUSPARSE_OPERATION_TRANSPOSE,
		N, nz, descr_L, d_ic, d_rowIdx, d_colIdx, info_U, &i_temp_buf_size));

	// if needs more buffer, free memory and re-allocate buffer
	if (i_temp_buf_size > u_temp_buf_size)
	{
		u_temp_buf_size = i_temp_buf_size;
	}

	// Compute buffer size for matrix-vector multiplicatyion
	error_check(cusparseSpMV_bufferSize(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, smat_A,
		dvec_p, &zero, dvec_q, CUDA_R_64F, CUSPARSE_CSRMV_ALG1, &u_temp_buf_size2));

	if (u_temp_buf_size2 > u_temp_buf_size)
	{
		u_temp_buf_size = u_temp_buf_size2;
	}

	error_check(cusparseSpMV_bufferSize(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, smat_A,
		dvec_x, &zero, dvec_q, CUDA_R_64F, CUSPARSE_CSRMV_ALG1, &u_temp_buf_size2));

	if (u_temp_buf_size2 > u_temp_buf_size)
	{
		u_temp_buf_size = u_temp_buf_size2;
	}


	// re-allocate buffer
	if (u_temp_buf_size > buf_size)
	{
		buf_size = u_temp_buf_size;
		cudaFree(d_buf);
		error_check(cudaMalloc(&d_buf, buf_size));
	}

	/* --- Initialize variables --- */

	// analysis phase
	error_check(cusparseDcsrsv2_analysis(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		N, nz, descr_L, d_ic, d_rowIdx, d_colIdx, info_L, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf));
	error_check(cusparseDcsrsv2_analysis(cusHandle, CUSPARSE_OPERATION_TRANSPOSE,
		N, nz, descr_L, d_ic, d_rowIdx, d_colIdx, info_U, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf));

	// x = 0
	error_check(cudaMemset(d_x, 0, N * sizeof(double)));
	// r0 = b  (since x == 0, b - A*x = b)
	error_check(cudaMemcpy(d_r, d_b, N * sizeof(double), cudaMemcpyDeviceToDevice));

	// set initial guess
	if(d_guess != NULL)
	{
		// x = guess
		error_check(cudaMemcpy(d_x, d_guess, N * sizeof(double), cudaMemcpyDeviceToDevice));
		// r0 = b - A*x
		//     q = A*x
		//     r0 = -q + b
		error_check(cusparseSpMV(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, smat_A,
			dvec_x, &zero, dvec_q, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, d_buf));
		double n_one = -1;
		error_check(cublasDaxpy(cubHandle, N, &n_one, d_q, 1, d_r, 1));
	}
	
	/* --- Perform conjugate gradient --- */

	// Solving linear system
	for (k = 0; k < max_iter; ++k)
	{
		// if ||rk|| < tolerance
		error_check(cublasDnrm2(cubHandle, N, d_r, 1, &rTr));
		//std::cout << "Iteration " << k << ": " << rTr << std::endl;
		if (rTr < tolerance)
		{
			break;
		}

		// Solve L*y = rk
		error_check(cusparseDcsrsv2_solve(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, nz, &one,
			descr_L, d_ic, d_rowIdx, d_colIdx, info_L, d_r, d_y, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf));

		// Solve L^T*zk = y
		error_check(cusparseDcsrsv2_solve(cusHandle, CUSPARSE_OPERATION_TRANSPOSE, N, nz, &one,
			descr_L, d_ic, d_rowIdx, d_colIdx, info_U, d_y, d_z, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_buf));

		// rho_t = r{k-1} * z{k-1}
		rho_t = rho;  
		// rho = rk * zk
		error_check(cublasDdot(cubHandle, N, d_r, 1, d_z, 1, &rho));

		if (k == 0)
		{
			// pk = zk
			error_check(cublasDcopy(cubHandle, N, d_z, 1, d_p, 1));
		}
		else
		{
			// beta = (rk*zk) / (r{k-1}*z{k-1})
			beta = rho / rho_t;
			// pk = zk + beta*p{k-1}
			error_check(cublasDscal(cubHandle, N, &beta, d_p, 1));
			error_check(cublasDaxpy(cubHandle, N, &one, d_z, 1, d_p, 1));
		}

		// q = A*pk
		error_check(cusparseSpMV(cusHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, smat_A,
			dvec_p, &zero, dvec_q, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, d_buf));

		// alpha = (rk*zk) / (pk*q)
		error_check(cublasDdot(cubHandle, N, d_p, 1, d_q, 1, &pTq));
		alpha = rho / pTq;

		// x{k+1} = xk + alpha*pk
		error_check(cublasDaxpy(cubHandle, N, &alpha, d_p, 1, d_x, 1));
		
		// r{k+1} = rk - alpha*q 
		double n_alpha = -alpha;
		error_check(cublasDaxpy(cubHandle, N, &n_alpha, d_q, 1, d_r, 1));
	}

	// free buffer
	cudaFree(d_buf);

	// free objects
	error_check(cusparseDestroySpMat(smat_A));
	error_check(cusparseDestroyDnVec(dvec_p));
	error_check(cusparseDestroyDnVec(dvec_q));
	error_check(cusparseDestroyDnVec(dvec_x));
	error_check(cusparseDestroyCsric02Info(icinfo_A));
	error_check(cusparseDestroyCsrsv2Info(info_L));
	error_check(cusparseDestroyCsrsv2Info(info_U));

	// return true if converged
	return rTr < tolerance;
}

double *ICCGsolver::x_ptr()
{
	return d_x;
}

double ICCGsolver::err()
{
	return rTr;
}

int ICCGsolver::iter_count()
{
	return k;
}

void ICCGsolver::check_and_resize(int N, int nz)
{
	// allocate N
	if (this->N < N)
	{
		this->N = N;
		free_array_memory();
		allocate_array_memory();
	}

	if (this->nz < nz)
	{
		this->nz = nz;
		free_nonzero_memory();
		allocate_nonzero_memory();
	}
}

void ICCGsolver::allocate_array_memory()
{
	error_check(cudaMalloc(&d_x, N * sizeof(double)));
	error_check(cudaMalloc(&d_y, N * sizeof(double)));
	error_check(cudaMalloc(&d_z, N * sizeof(double)));
	error_check(cudaMalloc(&d_r, N * sizeof(double)));
	error_check(cudaMalloc(&d_rt, N * sizeof(double)));
	error_check(cudaMalloc(&d_xt, N * sizeof(double)));
	error_check(cudaMalloc(&d_q, N * sizeof(double)));
	error_check(cudaMalloc(&d_p, N * sizeof(double)));
}

void ICCGsolver::allocate_nonzero_memory()
{
	error_check(cudaMalloc(&d_ic, nz * sizeof(double)));
}

void ICCGsolver::allocate_memory()
{
	allocate_array_memory();
	allocate_nonzero_memory();
}

void ICCGsolver::free_array_memory()
{
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);
	cudaFree(d_r);
	cudaFree(d_rt);
	cudaFree(d_xt);
	cudaFree(d_q);
	cudaFree(d_p);
}

void ICCGsolver::free_nonzero_memory()
{
	cudaFree(d_ic);
}

void ICCGsolver::free_memory()
{
	free_array_memory();
	free_nonzero_memory();
}

