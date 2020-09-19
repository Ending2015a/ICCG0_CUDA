#ifndef __ERROR_HANDLER_H__
#define __ERROR_HANDLER_H__

#include <iostream>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

namespace zex
{

// some simple meta programming

// ===== is_any =====

template<typename T, typename ... R>
struct is_any : std::false_type {};

template<typename T, typename F>
struct is_any<T, F> : std::is_same<T, F> {};

template<typename T, typename F, typename ...R>
struct is_any<T, F, R...> :
        std::integral_constant<bool, std::is_same<T, F>::value
                                    || is_any<T, R...>::value>{};


// ===== type_index =====

template<int _case, typename T, typename ...R>
struct _type_index : std::false_type {};

template<int _case, typename T, typename F>
struct _type_index<_case, T, F> : std::conditional<std::is_same<T, F>::value,
                                        std::integral_constant<int, _case>,
                                        std::integral_constant<int, -1>>::type{};

template<int _case, typename T, typename F, typename ...R>
struct _type_index<_case, T, F, R...> : std::conditional<std::is_same<T, F>::value,
                                        _type_index<_case, T, F>, _type_index<_case+1, T, R...>>::type{};

template<typename T, typename ...R>
struct type_index : _type_index<0, T, R...>{};



// ===== type_case =====

template<int _case, typename ...R>
struct _type_case : std::false_type{};

template<int _case, typename T>
struct _type_case<_case, T> {  typedef T type;  };

template<int _case, typename T, typename ...R>
struct _type_case<_case, T, R...> : std::conditional<_case==0, _type_case<_case, T>,
                                                               _type_case<_case-1, R...>>::type{};

template<int _case, typename ...T>
struct type_case : _type_case<_case, T...> {};


// ===== tv_pair =====

template<typename T, T V>
struct tv_pair {
    typedef T type;
    static constexpr T value=V;
};

// ===== pick/options =====

// recursive call
template<int _case>
struct pick{
    constexpr static bool _options() { return false; }

    template<typename T, typename ...Tail>
    constexpr static auto _options(const T& head, Tail&&...tail)
        -> decltype(pick<_case-1>::_options(std::forward<Tail>(tail)...))
    {
        return pick<_case-1>::_options(tail...);
    }

    template<typename T, typename ...Tail>
    constexpr static auto options(const T& head, Tail&&...tail)
        -> decltype(pick<_case-1>::_options(tail...))
    {
        return pick<_case-1>::_options(tail...);
    }
};

// stopping state
template<>
struct pick<0>{
    template<typename T, typename ...Tail>
    constexpr static auto _options(const T& head, Tail&&...tail)
        -> decltype(head)
    {
        return head;
    }

    template<typename T, typename ...Tail>
    constexpr static auto options(const T& head, Tail&&...tail)
        -> decltype(head)
    {
        return head;
    }
};

// namespace zex {end}
}


// =========================================


#define error_check(err) error_handler(err, #err, __LINE__, __FILE__)

#ifdef ERROR_HELPER_DONT_IMPLEMENT
const char *error_message(cublasStatus_t err);
const char *error_message(cusparseStatus_t err);
const char *error_message(cudaError_t err);
#else

const char *error_message(cublasStatus_t err)
{
    switch(err)
    {
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "library not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "failed to allocate resources";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "got invalid value";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "device does not support double/half precision";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "failed to launch on the GPU";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "access gpu error";
    }
    return "other error occurred";
}

const char *error_message(cusparseStatus_t err)
{
    switch(err)
    {
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "library not initialized";
        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "failed to allocate resources";
        case CUSPARSE_STATUS_INVALID_VALUE:
            return "got invalid value";
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "device does not support double/half precision";
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "failed to launch on the GPU";
		case CUSPARSE_STATUS_INTERNAL_ERROR:
			return "An internal cuSPARSE operation failed";
		case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
			return "The matrix type is not supported by this function";
		case CUSPARSE_STATUS_NOT_SUPPORTED:
			return "The operation or data type combination is currently not supported by the function";
		case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:
			return "Out of memory";

    }
    return "other error occurred";
}

const char *error_message(cudaError_t err)
{
    switch(err)
    {
        case cudaErrorMemoryAllocation:
            return "failed to allocate resources";
        case cudaErrorInitializationError:
            return "failed to initialize cuda";
        case cudaErrorInvalidValue:
            return "got invalid value";
    }
    return "other error occurred";
}

#endif

template<typename t, t v>
using pair = zex::tv_pair<t, v>;

// some simple metaprogramming, ensure that the error type is one of those types in is_any list
template<typename errT>
void error_handler(errT err, const std::string& func, const int& line, const std::string& file)
{
    static_assert(zex::is_any<errT, cudaError_t, cublasStatus_t, cusparseStatus_t>::value, 
                "Error type must be one of the types: cudaError_t, cublasStatus_t or cusparseStatue_t");

    // type switch-case
    using tidx = zex::type_index<errT, cudaError_t, cublasStatus_t, cusparseStatus_t>;

    // pick the "tidx::value"th option from the options list
    if( err != zex::pick<tidx::value>::options(cudaSuccess, CUBLAS_STATUS_SUCCESS, CUSPARSE_STATUS_SUCCESS))
    {
        // pick the "tidx::value"th option from the options list
        std::cout << zex::pick<tidx::value>::options("[cuda ERROR]", "[cuBLAS ERROR]", "[cuSPARSE ERROR]")
                  << " " << error_message(err) << std::endl
                  << "    in line [" << line << "] : " << func << std::endl
                  << "    in file " << file << std::endl;
        exit(1);
    }

}

#endif
