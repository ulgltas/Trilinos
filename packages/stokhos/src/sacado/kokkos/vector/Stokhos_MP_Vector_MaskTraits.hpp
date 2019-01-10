// @HEADER
// ***********************************************************************
//
//                           Stokhos Package
//                 Copyright (2009) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Eric T. Phipps (etphipp@sandia.gov).
//
// ***********************************************************************
// @HEADER

#ifndef STOKHOS_MP_VECTOR_MASKTRAITS_HPP
#define STOKHOS_MP_VECTOR_MASKTRAITS_HPP

//#define STOKHOS_MP_VECTOR_MASK_USE_II

#include "Stokhos_Sacado_Kokkos_MP_Vector.hpp"
#include <iostream>
#include <cmath>
//#include <tuple>
//#include <utility>
#include <initializer_list>

//#ifdef STOKHOS_MP_VECTOR_MASK_USE_II
#include <immintrin.h>
#define STOKHOS_MASK_AVX_VECTOR_SIZE 8

template<int n, typename T>
union fused_vector_ensemble_type {
  __m512d v[n/STOKHOS_MASK_AVX_VECTOR_SIZE] __attribute__((aligned(64)));
  T ensemble __attribute__((aligned(64)));
};

//#define M512D_ENSEMBLE_LOAD(ensemble,i) _mm512_load_pd(&((ensemble)[i*STOKHOS_MASK_AVX_VECTOR_SIZE]))
//#define M512D_ENSEMBLE_LOAD_2(ensemble,i) _mm512_load_pd(&(ensemble[i*STOKHOS_MASK_AVX_VECTOR_SIZE]))
//#define M512D_ENSEMBLE_STORE(ensemble,i,value) _mm512_store_pd(&((ensemble)[i*STOKHOS_MASK_AVX_VECTOR_SIZE]), value)
//#endif

template<typename S>
KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) __m512d M512D_ENSEMBLE_LOAD(const Sacado::MP::Vector<S>  &ensemble,const int &i)
{
  //const double * p = &(ensemble[i*STOKHOS_MASK_AVX_VECTOR_SIZE]);
  return _mm512_load_pd(&(ensemble[i*STOKHOS_MASK_AVX_VECTOR_SIZE]));
}

template<typename S>
KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) void M512D_ENSEMBLE_STORE(Sacado::MP::Vector<S>  & ensemble, const int &i, const __m512d &value)
{
  //double * p = &(ensemble[i*STOKHOS_MASK_AVX_VECTOR_SIZE]);
  _mm512_store_pd(&(ensemble[i*STOKHOS_MASK_AVX_VECTOR_SIZE]), value);
}
/*
template<>
KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) __m512d M512D_ENSEMBLE_LOAD<double>(const double &ensemble,const int &i)
{
  //static_assert(false, "M512D_ENSEMBLE_LOAD<double> want to be compiled");
  return _mm512_set1_pd(ensemble);
}

template<>
KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) void M512D_ENSEMBLE_STORE<double>(double & ensemble, const int &i, const __m512d &value)
{
  //static_assert(false, "M512D_ENSEMBLE_STORE<double> want to be compiled");
  double * p = &ensemble;
  _mm512_store_pd(p, value);
}
*/

template <typename T>
struct EnsembleTraits_m {
    static const int size = 1;
    typedef T value_type;
    static const value_type& coeff(const T& x, int i) { return x; }
    static value_type& coeff(T& x, int i) { return x; }
};

template <typename S>
struct EnsembleTraits_m< Sacado::MP::Vector<S> > {
    static const int size = S::static_size;
    typedef typename S::value_type value_type;
    static const value_type& coeff(const Sacado::MP::Vector<S>& x, int i) {
        return x.fastAccessCoeff(i);
    }
    static value_type& coeff(Sacado::MP::Vector<S>& x, int i) {
        return x.fastAccessCoeff(i);
    }
};



template<typename scalar> class MaskedAssign
{
private:
    scalar &data;
    bool m;

public:
    MaskedAssign(scalar &data_, bool m_) : data(data_), m(m_) {};

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator = (const scalar & KOKKOS_RESTRICT s)
    {
        if(m)
            data = s;

        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator = (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        auto st_array = st.begin();

        if(m)
            data = st_array[0];
        else
            data = st_array[1];
        return *this;
    }


    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator += (const scalar & KOKKOS_RESTRICT s)
    {
        if(m)
            data += s;

        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator += (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        auto st_array = st.begin();

        if(m)
            data = st_array[0]+st_array[1];
        else
            data = st_array[2];
        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator -= (const scalar & KOKKOS_RESTRICT s)
    {
        if(m)
            data -= s;

        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator -= (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        auto st_array = st.begin();

        if(m)
            data = st_array[0]-st_array[1];
        else
            data = st_array[2];
        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator *= (const scalar & KOKKOS_RESTRICT s)
    {
        if(m)
            data *= s;

        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator *= (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        auto st_array = st.begin();

        if(m)
            data = st_array[0]*st_array[1];
        else
            data = st_array[2];
        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator /= (const scalar & KOKKOS_RESTRICT s)
    {
        if(m)
            data /= s;

        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator /= (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        auto st_array = st.begin();

        if(m)
            data = st_array[0]/st_array[1];
        else
            data = st_array[2];
        return *this;
    }
};



template<typename scalar> class Mask;
/*
template<typename scalar> class MaskedAssign
{
private:
    static const int size = EnsembleTraits_m<scalar>::size;
    scalar &data;
    Mask<scalar> m;

public:
    MaskedAssign(scalar &data_, Mask<scalar> m_) : data(data_), m(m_) {};

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator = (const scalar & KOKKOS_RESTRICT s)
    {
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
        typedef EnsembleTraits_m<scalar> ET;
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) = ET::coeff(s,i);
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_blend_pd(m.data[i],
                                                             M512D_ENSEMBLE_LOAD(data,i),
                                                             M512D_ENSEMBLE_LOAD(s,i)));
#endif
      return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator = (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        auto st_array = st.begin();
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
        typedef EnsembleTraits_m<scalar> ET;
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) = ET::coeff(st_array[0],i);
            else
                ET::coeff(data,i) = ET::coeff(st_array[1],i);
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_blend_pd(m.data[i],
                                                             M512D_ENSEMBLE_LOAD(st_array[1],i),
                                                             M512D_ENSEMBLE_LOAD(st_array[0],i)));
      
#endif
        return *this;
    }


    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator += (const scalar & KOKKOS_RESTRICT s)
    {
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
        typedef EnsembleTraits_m<scalar> ET;
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) += ET::coeff(s,i);
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_add_pd(M512D_ENSEMBLE_LOAD(data,i),
                                                           m.data[i],
                                                           M512D_ENSEMBLE_LOAD(data,i),
                                                           M512D_ENSEMBLE_LOAD(s,i)));
#endif
        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator += (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        auto st_array = st.begin();
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
        typedef EnsembleTraits_m<scalar> ET;
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) = ET::coeff(st_array[0],i)+ET::coeff(st_array[1],i);
            else
                ET::coeff(data,i) = ET::coeff(st_array[2],i);
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_add_pd(M512D_ENSEMBLE_LOAD(st_array[2],i),
                                                           m.data[i],
                                                           M512D_ENSEMBLE_LOAD(st_array[0],i),
                                                           M512D_ENSEMBLE_LOAD(st_array[1],i)));
#endif
        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator -= (const scalar & KOKKOS_RESTRICT s)
    {
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
        typedef EnsembleTraits_m<scalar> ET;
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) -= ET::coeff(s,i);
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_sub_pd(M512D_ENSEMBLE_LOAD(data,i),
                                                           m.data[i],
                                                           M512D_ENSEMBLE_LOAD(data,i),
                                                           M512D_ENSEMBLE_LOAD(s,i)));
#endif
        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator -= (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        auto st_array = st.begin();
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
        typedef EnsembleTraits_m<scalar> ET;
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) = ET::coeff(st_array[0],i)-ET::coeff(st_array[1],i);
            else
                ET::coeff(data,i) = ET::coeff(st_array[2],i);
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_sub_pd(M512D_ENSEMBLE_LOAD(st_array[2],i),
                                                           m.data[i],
                                                           M512D_ENSEMBLE_LOAD(st_array[0],i),
                                                           M512D_ENSEMBLE_LOAD(st_array[1],i)));
#endif
        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator *= (const scalar & KOKKOS_RESTRICT s)
    {
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
        typedef EnsembleTraits_m<scalar> ET;
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) *= ET::coeff(s,i);
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_mul_pd(M512D_ENSEMBLE_LOAD(data,i),
                                                           m.data[i],
                                                           M512D_ENSEMBLE_LOAD(data,i),
                                                           M512D_ENSEMBLE_LOAD(s,i)));
#endif
        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator *= (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        auto st_array = st.begin();
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
        typedef EnsembleTraits_m<scalar> ET;
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) = ET::coeff(st_array[0],i)*ET::coeff(st_array[1],i);
            else
                ET::coeff(data,i) = ET::coeff(st_array[2],i);
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_mul_pd(M512D_ENSEMBLE_LOAD(st_array[2],i),
                                                           m.data[i],
                                                           M512D_ENSEMBLE_LOAD(st_array[0],i),
                                                           M512D_ENSEMBLE_LOAD(st_array[1],i)));
#endif
        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator /= (const scalar & KOKKOS_RESTRICT s)
    {
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
        typedef EnsembleTraits_m<scalar> ET;
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) /= ET::coeff(s,i);
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_div_pd(M512D_ENSEMBLE_LOAD(data,i),
                                                           m.data[i],
                                                           M512D_ENSEMBLE_LOAD(data,i),
                                                           M512D_ENSEMBLE_LOAD(s,i)));
#endif
        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator /= (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        auto st_array = st.begin();
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
        typedef EnsembleTraits_m<scalar> ET;
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) = ET::coeff(st_array[0],i)/ET::coeff(st_array[1],i);
            else
                ET::coeff(data,i) = ET::coeff(st_array[2],i);
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_div_pd(M512D_ENSEMBLE_LOAD(st_array[2],i),
                                                           m.data[i],
                                                           M512D_ENSEMBLE_LOAD(st_array[0],i),
                                                           M512D_ENSEMBLE_LOAD(st_array[1],i)));
#endif
        return *this;
    }
};
*/

template<typename S> class MaskedAssign< Sacado::MP::Vector<S> >
{
    typedef Sacado::MP::Vector<S> scalar;
private:
    static const int size = S::static_size;
    scalar &data;
    Mask<scalar> m;

public:
    MaskedAssign(scalar &data_, Mask<scalar> m_) : data(data_), m(m_) {};

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator = (const scalar & KOKKOS_RESTRICT s)
    {
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.data[i])
                data[i] = s[i];
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_blend_pd(m.data[i],
                                                             M512D_ENSEMBLE_LOAD(data,i),
                                                             M512D_ENSEMBLE_LOAD(s,i)));
#endif
      return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator = (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        auto st_array = st.begin();
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.data[i])
                data[i] = st_array[0][i];
            else
                data[i] = st_array[1][i];
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_blend_pd(m.data[i],
                                                             M512D_ENSEMBLE_LOAD(st_array[1],i),
                                                             M512D_ENSEMBLE_LOAD(st_array[0],i)));
      
#endif
        return *this;
    }


    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator += (const scalar & KOKKOS_RESTRICT s)
    {
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.data[i])
                data[i] += s[i];
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_add_pd(M512D_ENSEMBLE_LOAD(data,i),
                                                           m.data[i],
                                                           M512D_ENSEMBLE_LOAD(data,i),
                                                           M512D_ENSEMBLE_LOAD(s,i)));
#endif
        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator += (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        auto st_array = st.begin();
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.data[i])
                data[i] = st_array[0][i]+st_array[1][i];
            else
                data[i] = st_array[2][i];
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_add_pd(M512D_ENSEMBLE_LOAD(st_array[2],i),
                                                           m.data[i],
                                                           M512D_ENSEMBLE_LOAD(st_array[0],i),
                                                           M512D_ENSEMBLE_LOAD(st_array[1],i)));
#endif
        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator -= (const scalar & KOKKOS_RESTRICT s)
    {
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.data[i])
                data[i] -= s[i];
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_sub_pd(M512D_ENSEMBLE_LOAD(data,i),
                                                           m.data[i],
                                                           M512D_ENSEMBLE_LOAD(data,i),
                                                           M512D_ENSEMBLE_LOAD(s,i)));
#endif
        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator -= (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        auto st_array = st.begin();
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.data[i])
                data[i] = st_array[0][i]-st_array[1][i];
            else
                data[i] = st_array[2][i];
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_sub_pd(M512D_ENSEMBLE_LOAD(st_array[2],i),
                                                           m.data[i],
                                                           M512D_ENSEMBLE_LOAD(st_array[0],i),
                                                           M512D_ENSEMBLE_LOAD(st_array[1],i)));
#endif
        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator *= (const scalar & KOKKOS_RESTRICT s)
    {
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.data[i])
                data[i] *= s[i];
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_mul_pd(M512D_ENSEMBLE_LOAD(data,i),
                                                           m.data[i],
                                                           M512D_ENSEMBLE_LOAD(data,i),
                                                           M512D_ENSEMBLE_LOAD(s,i)));
#endif
        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator *= (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        auto st_array = st.begin();
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.data[i])
                data[i] = st_array[0][i]*st_array[1][i];
            else
                data[i] = st_array[2][i];
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_mul_pd(M512D_ENSEMBLE_LOAD(st_array[2],i),
                                                           m.data[i],
                                                           M512D_ENSEMBLE_LOAD(st_array[0],i),
                                                           M512D_ENSEMBLE_LOAD(st_array[1],i)));
#endif
        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator /= (const scalar & KOKKOS_RESTRICT s)
    {
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.data[i])
                data[i] /= s[i];
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_div_pd(M512D_ENSEMBLE_LOAD(data,i),
                                                           m.data[i],
                                                           M512D_ENSEMBLE_LOAD(data,i),
                                                           M512D_ENSEMBLE_LOAD(s,i)));
#endif
        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator /= (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        auto st_array = st.begin();
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.data[i])
                data[i] = st_array[0][i]/st_array[1][i];
            else
                data[i] = st_array[2][i];
#else
#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<m.size_uc; ++i)
            M512D_ENSEMBLE_STORE(data,i,_mm512_mask_div_pd(M512D_ENSEMBLE_LOAD(st_array[2],i),
                                                           m.data[i],
                                                           M512D_ENSEMBLE_LOAD(st_array[0],i),
                                                           M512D_ENSEMBLE_LOAD(st_array[1],i)));
#endif
        return *this;
    }
};

template<typename scalar> class Mask
{
private:
    static const int size = EnsembleTraits_m<scalar>::size;
public:
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
    bool data[size] __attribute__((aligned(64)));
#else
    static const int size_uc = (size == 1 ? 1 : size/STOKHOS_MASK_AVX_VECTOR_SIZE);
    unsigned char data[size_uc] __attribute__((aligned(64)));
#endif


public:
    Mask(){
    #ifndef STOKHOS_MP_VECTOR_MASK_USE_II
        for(int i=0; i<size; ++i)
            this->set(i,false);
    #else
    #endif
    }

    Mask(bool a){
        for(int i=0; i<size; ++i)
            this->set(i,a);
    }
  
    Mask(const Mask &a){
    #ifndef STOKHOS_MP_VECTOR_MASK_USE_II
    for(int i=0; i<size; ++i)
      data[i] = a.data[i];
    #else
    for(int i=0; i<size_uc; ++i)
      data[i] = a.data[i];
    #endif
    }

    int getSize() const {return size;}

    bool operator> (double v)
    {
        double sum = 0;
        for(int i=0; i<size; ++i)
            sum = sum + this->get(i);

        return sum > v*size;
    }

    bool operator< (double v)
    {
        double sum = 0;
        for(int i=0; i<size; ++i)
            sum = sum + this->get(i);

        return sum < v*size;
    }

    bool operator>= (double v)
    {
        double sum = 0;
        for(int i=0; i<size; ++i)
            sum = sum + this->get(i);

        return sum >= v*size;
    }

    bool operator<= (double v)
    {
        double sum = 0;
        for(int i=0; i<size; ++i)
            sum = sum + this->get(i);

        return sum <= v*size;
    }

    bool operator== (double v)
    {
        double sum = 0;
        for(int i=0; i<size; ++i)
            sum = sum + this->get(i);

        return sum == v*size;
    }

    bool operator!= (double v)
    {
        double sum = 0;
        for(int i=0; i<size; ++i)
            sum = sum + this->get(i);

        return sum != v*size;
    }

    bool operator== (const Mask<scalar> &m2)
    {
        bool all = true;
        for (int i = 0; i < size; ++i) {
            all && (this->get(i) == m2.get(i));
        }
        return all;
    }

    bool operator!= (const Mask<scalar> &m2)
    {
        return !(this==m2);
    }

    Mask<scalar> operator&& (const Mask<scalar> &m2)
    {
        Mask<scalar> m3;
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
        for(int i=0; i<size; ++i)
            m3.set(i,(this->get(i) && m2.get(i)));
#else
        for(int i=0; i<size_uc; ++i)
            m3.data[i]=_mm512_kand(this->data[i],m2.data[i]);
#endif
        return m3;
    }

    Mask<scalar> operator|| (const Mask<scalar> &m2)
    {
        Mask<scalar> m3;
        for(int i=0; i<size; ++i)
            m3.set(i,(this->get(i) || m2.get(i)));

        return m3;
    }

    Mask<scalar> operator&& (bool m2)
    {
        Mask<scalar> m3;
        for(int i=0; i<size; ++i)
            m3.set(i,(this->get(i) && m2));
        return m3;
    }

    Mask<scalar> operator|| (bool m2)
    {
        Mask<scalar> m3;
        for(int i=0; i<size; ++i)
            m3.set(i,(this->get(i) || m2));

        return m3;
    }

    Mask<scalar> operator+ (const Mask<scalar> &m2)
    {
        Mask<scalar> m3;
        for(int i=0; i<size; ++i)
            m3.set(i,(this->get(i) + m2.get(i)));

        return m3;
    }

    Mask<scalar> operator- (const Mask<scalar> &m2)
    {
        Mask<scalar> m3;
        for(int i=0; i<size; ++i)
            m3.set(i,(this->get(i) - m2.get(i)));

        return m3;
    }

    scalar operator* (const scalar &v)
    {
        typedef EnsembleTraits_m<scalar> ET;
        scalar v2;
        for(int i=0; i<size; ++i)
            ET::coeff(v2,i) = ET::coeff(v,i)*this->get(i);

        return v2;
    }

#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) bool get (int i) const
    {
        return this->data[i];
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) void set (int i, bool b)
    {
        this->data[i] = b;
    }
#else
    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) bool get (int i) const
    {
        //static_assert(false, "get want to be compiled");
        int j = i/STOKHOS_MASK_AVX_VECTOR_SIZE;
        return (this->data[j] & (1 << i%STOKHOS_MASK_AVX_VECTOR_SIZE)) ? true : false;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) void set (int i, bool b)
    {
        //static_assert(false, "set want to be compiled");
        int j = i/STOKHOS_MASK_AVX_VECTOR_SIZE;
        if(b)
          this->data[j] |= 0x01 << i%STOKHOS_MASK_AVX_VECTOR_SIZE;
        else
          this->data[j] &= ~(0x01 << i%STOKHOS_MASK_AVX_VECTOR_SIZE);
    }
#endif

    Mask<scalar> operator! ()
    {
        Mask<scalar> m2;
#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
        for(int i=0; i<size; ++i)
            m2.set(i,!(this->get(i)));
#else
        for(int i=0; i<size_uc; ++i)
            m2.data[i]=_mm512_knot(this->data[i]);
#endif
        return m2;
    }

    operator bool() const
    {
        return this->get(0);
    }

    operator double() const
    {
        double sum = 0;
        for(int i=0; i<size; ++i)
            sum = sum + this->get(i);

        return sum/size;
    }
};

template<typename scalar> KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar> mask_assign(bool b, scalar *s)
{
    Mask<scalar> m = Mask<scalar>(b);
    MaskedAssign<scalar> maskedAssign = MaskedAssign<scalar>(*s,m);
    return maskedAssign;
}

template<typename scalar> KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar> mask_assign(Mask<scalar> m, scalar *s)
{
    MaskedAssign<scalar> maskedAssign = MaskedAssign<scalar>(*s,m);
    return maskedAssign;
}

template<typename scalar> KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar> mask_assign(bool b, scalar &s)
{
    Mask<scalar> m = Mask<scalar>(b);
    MaskedAssign<scalar> maskedAssign = MaskedAssign<scalar>(s,m);
    return maskedAssign;
}

template<typename scalar> KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar> mask_assign(Mask<scalar> m, scalar &s)
{
    MaskedAssign<scalar> maskedAssign = MaskedAssign<scalar>(s,m);
    return maskedAssign;
}

template<typename scalar> KOKKOS_INLINE_FUNCTION std::ostream &operator<<(std::ostream &os, const Mask<scalar>& m) {
    os << "[ ";
    for(int i=0; i<m.getSize(); ++i)
        os << m.get(i) << " ";
    return os << "]";
}

template<typename S> KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) Sacado::MP::Vector<S> operator* (const Sacado::MP::Vector<S> &a1, const Mask<Sacado::MP::Vector<S>> &m)
{
    typedef EnsembleTraits_m<Sacado::MP::Vector<S>> ET;
    Sacado::MP::Vector<S> mul;
#pragma vector aligned
#pragma ivdep
#pragma unroll
    for(int i=0; i<ET::size; ++i){
        ET::coeff(mul,i) = ET::coeff(a1,i)*m.get(i);
    }
    return mul;
}

template<typename S> KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) Sacado::MP::Vector<S> operator* (const typename S::value_type &a1, const Mask<Sacado::MP::Vector<S>> &m)
{
    Sacado::MP::Vector<S> mul;
    typedef EnsembleTraits_m<Sacado::MP::Vector<S>> ET;
#pragma vector aligned
#pragma ivdep
#pragma unroll
    for(int i=0; i<ET::size; ++i){
        ET::coeff(mul,i) = m.get(i)*a1;
    }
    return mul;
}

template<typename S> KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) Sacado::MP::Vector<S> operator* (const Mask<Sacado::MP::Vector<S>> &m, const typename S::value_type &a1)
{
    Sacado::MP::Vector<S> mul;
    typedef EnsembleTraits_m<Sacado::MP::Vector<S>> ET;
#pragma vector aligned
#pragma ivdep
#pragma unroll
    for(int i=0; i<ET::size; ++i){
        ET::coeff(mul,i) = m.get(i)*a1;
    }
    return mul;
}

namespace Sacado {
    namespace MP {
        template <typename S> Vector<S> copysign(const Vector<S> &a1, const Vector<S> &a2)
        {
            typedef EnsembleTraits_m< Vector<S> > ET;

            Vector<S> a_out;

            using std::copysign;
            for(int i=0; i<ET::size; ++i){
                ET::coeff(a_out,i) = copysign(ET::coeff(a1,i),ET::coeff(a2,i));
            }
            return a_out;
        }
    }
}


template<typename S> Mask<Sacado::MP::Vector<S> > signbit_v(const Sacado::MP::Vector<S> &a1)
{
    typedef EnsembleTraits_m<Sacado::MP::Vector<S> > ET;
    using std::signbit;

    Mask<Sacado::MP::Vector<S> > mask;
#pragma vector aligned
#pragma ivdep
#pragma unroll
    for(int i=0; i<ET::size; ++i)
        mask.set(i, signbit(ET::coeff(a1,i)));
    return mask;
}


#define OPNAME ==
#define imm8 16
#include "Stokhos_MP_Vector_MaskTraits_rel_ops_tmpl.hpp"
#undef OPNAME
#undef imm8

#define OPNAME !=
#define imm8 28
#include "Stokhos_MP_Vector_MaskTraits_rel_ops_tmpl.hpp"
#undef OPNAME
#undef imm8

#define OPNAME >
#define imm8 14
#include "Stokhos_MP_Vector_MaskTraits_rel_ops_tmpl.hpp"
#undef OPNAME
#undef imm8

#define OPNAME >=
#define imm8 13
#include "Stokhos_MP_Vector_MaskTraits_rel_ops_tmpl.hpp"
#undef OPNAME
#undef imm8

#define OPNAME <
#define imm8 1
#include "Stokhos_MP_Vector_MaskTraits_rel_ops_tmpl.hpp"
#undef OPNAME
#undef imm8

#define OPNAME <=
#define imm8 2
#include "Stokhos_MP_Vector_MaskTraits_rel_ops_tmpl.hpp"
#undef OPNAME
#undef imm8


/*
MP_VECTOR_RELOP_MACRO(==,16) //_CMP_EQ_OS
MP_VECTOR_RELOP_MACRO(!=,28) //_CMP_NEQ_OS
MP_VECTOR_RELOP_MACRO(>,14) //_CMP_GT_OS
MP_VECTOR_RELOP_MACRO(>=,13) //_CMP_GE_OS
MP_VECTOR_RELOP_MACRO(<,1) //_CMP_LT_OS
MP_VECTOR_RELOP_MACRO(<=,2) //_CMP_LE_OS
*/



#define MP_EXPR_RELOP_MACRO(OP)                                         \
namespace Sacado {                                                      \
  namespace MP {                                                        \
                                                                        \
    template <typename V, typename V2>                                  \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask<V>                                                             \
    operator OP (const Expr<V> &a1,                                     \
                 const Expr<V2> &a2)                                    \
    {                                                                   \
      const V& v1 = a1.derived();                                       \
      const V2& v2 = a2.derived();                                      \
      Mask<V> mask;                                                     \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      if (v2.hasFastAccess(v1.size())) {                                \
        for(int i=0; i<v1.size(); ++i)                                  \
          mask.set(i, v1.fastAccessCoeff(i) OP v2.fastAccessCoeff(i));  \
      }                                                                 \
      else{                                                             \
        for(int i=0; i<v1.size(); ++i)                                  \
          mask.set(i, v1.fastAccessCoeff(i) OP v2.coeff(i));            \
      }                                                                 \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename V, typename V2>                                  \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask<V>                                                             \
    operator OP (const volatile Expr<V> &a1,                            \
                 const volatile Expr<V2> &a2)                           \
    {                                                                   \
      const volatile V& v1 = a1.derived();                              \
      const volatile V2& v2 = a2.derived();                             \
      Mask<V> mask;                                                     \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      if (v2.hasFastAccess(v1.size())) {                                \
        for(int i=0; i<v1.size(); ++i)                                  \
          mask.set(i, v1.fastAccessCoeff(i) OP v2.fastAccessCoeff(i));  \
      }                                                                 \
      else{                                                             \
        for(int i=0; i<v1.size(); ++i)                                  \
          mask.set(i, v1.fastAccessCoeff(i) OP v2.coeff(i));            \
      }                                                                 \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename V, typename V2>                                  \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask<V>                                                             \
    operator OP (const Expr<V> &a1,                                     \
                 const volatile Expr<V2> &a2)                           \
    {                                                                   \
      const V& v1 = a1.derived();                                       \
      const volatile V2& v2 = a2.derived();                             \
      Mask<V> mask;                                                     \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      if (v2.hasFastAccess(v1.size())) {                                \
        for(int i=0; i<v1.size(); ++i)                                  \
          mask.set(i, v1.fastAccessCoeff(i) OP v2.fastAccessCoeff(i));  \
      }                                                                 \
      else{                                                             \
        for(int i=0; i<v1.size(); ++i)                                  \
          mask.set(i, v1.fastAccessCoeff(i) OP v2.coeff(i));            \
      }                                                                 \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename V, typename V2>                                  \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask<V>                                                             \
    operator OP (const volatile Expr<V> &a1,                            \
                 const Expr<V2> &a2)                                    \
    {                                                                   \
      const volatile V& v1 = a1.derived();                              \
      const V2& v2 = a2.derived();                                      \
      Mask<V> mask;                                                     \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      if (v2.hasFastAccess(v1.size())) {                                \
        for(int i=0; i<v1.size(); ++i)                                  \
          mask.set(i, v1.fastAccessCoeff(i) OP v2.fastAccessCoeff(i));  \
      }                                                                 \
      else{                                                             \
        for(int i=0; i<v1.size(); ++i)                                  \
          mask.set(i, v1.fastAccessCoeff(i) OP v2.coeff(i));            \
      }                                                                 \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename V>                                               \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask<V>                                                             \
    operator OP (const Expr<V> &a1,                                     \
                 const typename V::value_type &a2)                      \
    {                                                                   \
      const V& v1 = a1.derived();                                       \
      Mask<V> mask;                                                     \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      for(int i=0; i<v1.size(); ++i)                                    \
        mask.set(i, v1.fastAccessCoeff(i) OP a2);                       \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename V>                                               \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask<V>                                                             \
    operator OP (const volatile Expr<V> &a1,                            \
                 const typename V::value_type &a2)                      \
    {                                                                   \
      const volatile V& v1 = a1.derived();                              \
      Mask<V> mask;                                                     \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      for(int i=0; i<v1.size(); ++i)                                    \
        mask.set(i, v1.fastAccessCoeff(i) OP a2);                       \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename V>                                               \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask<V>                                                             \
    operator OP (const typename V::value_type &a1,                      \
                 const Expr<V> &a2)                                     \
    {                                                                   \
      const V& v2 = a2.derived();                                       \
      Mask<V> mask;                                                     \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      for(int i=0; i<v2.size(); ++i)                                    \
        mask.set(i, a1 OP v2.fastAccessCoeff(i));                       \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename V>                                               \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask<V>                                                             \
    operator OP (const typename V::value_type &a1,                      \
                 const volatile Expr<V> &a2)                            \
    {                                                                   \
      const volatile V& v2 = a2.derived();                              \
      Mask<V> mask;                                                     \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      for(int i=0; i<v1.size(); ++i)                                    \
        mask.set(i, a1 OP v2.fastAccessCoeff(i));                       \
      return mask;                                                      \
    }                                                                   \
  }                                                                     \
}

MP_EXPR_RELOP_MACRO(==)
MP_EXPR_RELOP_MACRO(!=)
MP_EXPR_RELOP_MACRO(<)
MP_EXPR_RELOP_MACRO(>)
MP_EXPR_RELOP_MACRO(<=)
MP_EXPR_RELOP_MACRO(>=)
MP_EXPR_RELOP_MACRO(<<=)
MP_EXPR_RELOP_MACRO(>>=)
MP_EXPR_RELOP_MACRO(&)
MP_EXPR_RELOP_MACRO(|)

#undef MP_EXPR_RELOP_MACRO


#if STOKHOS_USE_MP_VECTOR_SFS_SPEC

#ifndef STOKHOS_MP_VECTOR_MASK_USE_II

#define MP_SFS_RELOP_MACRO(OP)                                          \
namespace Sacado {                                                      \
  namespace MP {                                                        \
                                                                        \
    template <typename O, typename T, int N, typename D>                \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > >              \
    operator OP (const Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& a, \
                 const Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& b) \
    {                                                                   \
      Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > mask;      \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      for(int i=0; i<a.size(); ++i)                                     \
          mask.set(i, a.fastAccessCoeff(i) OP b.fastAccessCoeff(i));    \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename O, typename T, int N, typename D>                \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > >              \
    operator OP (const volatile Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& a, \
                 const Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& b) \
    {                                                                   \
      Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > mask;      \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      for(int i=0; i<a.size(); ++i)                                     \
          mask.set(i, a.fastAccessCoeff(i) OP b.fastAccessCoeff(i));    \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename O, typename T, int N, typename D>                \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > >              \
    operator OP (const Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& a, \
                 const volatile Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& b) \
    {                                                                   \
      Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > mask;      \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      for(int i=0; i<a.size(); ++i)                                     \
          mask.set(i, a.fastAccessCoeff(i) OP b.fastAccessCoeff(i));    \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename O, typename T, int N, typename D>                \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > >              \
    operator OP (const volatile Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& a, \
                 const volatile Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& b) \
    {                                                                   \
      Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > mask;      \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      for(int i=0; i<a.size(); ++i)                                     \
          mask.set(i, a.fastAccessCoeff(i) OP b.fastAccessCoeff(i));    \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename O, typename T, int N, typename D>                \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > >              \
    operator OP (const typename Vector< Stokhos::StaticFixedStorage<O,T,N,D> >::value_type& a, \
                 const Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& b) \
    {                                                                   \
      Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > mask;      \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      for(int i=0; i<b.size(); ++i)                                     \
          mask.set(i, a OP b.fastAccessCoeff(i));                       \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename O, typename T, int N, typename D>                \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > >              \
    operator OP (const typename Vector< Stokhos::StaticFixedStorage<O,T,N,D> >::value_type& a, \
                 const volatile Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& b) \
    {                                                                   \
      Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > mask;      \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      for(int i=0; i<b.size(); ++i)                                     \
          mask.set(i, a OP b.fastAccessCoeff(i));                       \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename O, typename T, int N, typename D>                \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > >              \
    operator OP (const Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& a, \
                 const typename Vector< Stokhos::StaticFixedStorage<O,T,N,D> >::value_type& b) \
    {                                                                   \
      Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > mask;      \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      for(int i=0; i<a.size(); ++i)                                     \
          mask.set(i, a.fastAccessCoeff(i) OP b);                       \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename O, typename T, int N, typename D>                \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > >              \
    operator OP (const volatile Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& a, \
                 const typename Vector< Stokhos::StaticFixedStorage<O,T,N,D> >::value_type& b) \
    {                                                                   \
      Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > mask;      \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      for(int i=0; i<a.size(); ++i)                                     \
          mask.set(i, a.fastAccessCoeff(i) OP b);                       \
      return mask;                                                      \
    }                                                                   \
  }                                                                     \
}

MP_SFS_RELOP_MACRO(==)
MP_SFS_RELOP_MACRO(!=)
MP_SFS_RELOP_MACRO(<)
MP_SFS_RELOP_MACRO(>)
MP_SFS_RELOP_MACRO(<=)
MP_SFS_RELOP_MACRO(>=)
MP_SFS_RELOP_MACRO(<<=)
MP_SFS_RELOP_MACRO(>>=)
MP_SFS_RELOP_MACRO(&)
MP_SFS_RELOP_MACRO(|)

#undef MP_SFS_RELOP_MACRO

#else

#define MP_SFS_RELOP_MACRO(OP,imm8)                                     \
namespace Sacado {                                                      \
  namespace MP {                                                        \
                                                                        \
    template <typename O, typename T, int N, typename D>                \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > >              \
    operator OP (const Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& a, \
                 const Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& b) \
    {                                                                   \
      typedef fused_vector_ensemble_type<Stokhos::StaticFixedStorage<O,T,N,D>::static_size,Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > FVET; \
      FVET a1_ii, a2_ii;                                                \
      Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > mask;      \
      a1_ii.ensemble = a;                                               \
      a2_ii.ensemble = b;                                               \
      for(int i=0; i<mask.size_uc; ++i)                                 \
        mask.data[i] = _mm512_cmp_pd_mask(a1_ii.v[i],a2_ii.v[i],imm8);  \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename O, typename T, int N, typename D>                \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > >              \
    operator OP (const volatile Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& a, \
                 const Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& b) \
    {                                                                   \
      typedef fused_vector_ensemble_type<Stokhos::StaticFixedStorage<O,T,N,D>::static_size,Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > FVET; \
      FVET a1_ii, a2_ii;                                                \
      Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > mask;      \
      a1_ii.ensemble = a;                                               \
      a2_ii.ensemble = b;                                               \
      for(int i=0; i<mask.size_uc; ++i)                                 \
        mask.data[i] = _mm512_cmp_pd_mask(a1_ii.v[i],a2_ii.v[i],imm8);  \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename O, typename T, int N, typename D>                \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > >              \
    operator OP (const Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& a, \
                 const volatile Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& b) \
    {                                                                   \
      typedef fused_vector_ensemble_type<Stokhos::StaticFixedStorage<O,T,N,D>::static_size,Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > FVET; \
      FVET a1_ii, a2_ii;                                                \
      Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > mask;      \
      a1_ii.ensemble = a;                                               \
      a2_ii.ensemble = b;                                               \
      for(int i=0; i<mask.size_uc; ++i)                                 \
        mask.data[i] = _mm512_cmp_pd_mask(a1_ii.v[i],a2_ii.v[i],imm8);  \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename O, typename T, int N, typename D>                \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > >              \
    operator OP (const volatile Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& a, \
                 const volatile Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& b) \
    {                                                                   \
      typedef fused_vector_ensemble_type<Stokhos::StaticFixedStorage<O,T,N,D>::static_size,Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > FVET; \
      FVET a1_ii, a2_ii;                                                \
      Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > mask;      \
      a1_ii.ensemble = a;                                               \
      a2_ii.ensemble = b;                                               \
      for(int i=0; i<mask.size_uc; ++i)                                 \
        mask.data[i] = _mm512_cmp_pd_mask(a1_ii.v[i],a2_ii.v[i],imm8);  \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename O, typename T, int N, typename D>                \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > >              \
    operator OP (const double & a,                                      \
                 const Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& b) \
    {                                                                   \
      typedef fused_vector_ensemble_type<Stokhos::StaticFixedStorage<O,T,N,D>::static_size,Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > FVET; \
      FVET a2_ii;                                                       \
      Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > mask;      \
      __m512d a1_ii = _mm512_set1_pd(a);                                \
      a2_ii.ensemble = b;                                               \
      for(int i=0; i<mask.size_uc; ++i)                                 \
        mask.data[i] = _mm512_cmp_pd_mask(a1_ii,a2_ii.v[i],imm8);       \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename O, typename T, int N, typename D>                \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > >              \
    operator OP (const double& a,                                       \
                 const volatile Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& b) \
    {                                                                   \
      typedef fused_vector_ensemble_type<Stokhos::StaticFixedStorage<O,T,N,D>::static_size,Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > FVET; \
      FVET a2_ii;                                                       \
      Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > mask;      \
      __m512d a1_ii = _mm512_set1_pd(a);                                \
      a2_ii.ensemble = b;                                               \
      for(int i=0; i<mask.size_uc; ++i)                                 \
        mask.data[i] = _mm512_cmp_pd_mask(a1_ii,a2_ii.v[i],imm8);       \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename O, typename T, int N, typename D>                \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > >              \
    operator OP (const Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& a, \
                 const double& b)                                       \
    {                                                                   \
      typedef fused_vector_ensemble_type<Stokhos::StaticFixedStorage<O,T,N,D>::static_size,Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > FVET; \
      FVET a1_ii;                                                       \
      Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > mask;      \
      __m512d a2_ii = _mm512_set1_pd(b);                                \
      a1_ii.ensemble = a;                                               \
      for(int i=0; i<mask.size_uc; ++i)                                 \
        mask.data[i] = _mm512_cmp_pd_mask(a1_ii.v[i],a2_ii,imm8);       \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename O, typename T, int N, typename D>                \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > >              \
    operator OP (const volatile Vector< Stokhos::StaticFixedStorage<O,T,N,D> >& a, \
                 const double& b)                                       \
    {                                                                   \
      typedef fused_vector_ensemble_type<Stokhos::StaticFixedStorage<O,T,N,D>::static_size,Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > FVET; \
      FVET a1_ii;                                                       \
      Mask< Vector< Stokhos::StaticFixedStorage<O,T,N,D> > > mask;      \
      __m512d a2_ii = _mm512_set1_pd(b);                                \
      a1_ii.ensemble = a;                                               \
      for(int i=0; i<mask.size_uc; ++i)                                 \
        mask.data[i] = _mm512_cmp_pd_mask(a1_ii.v[i],a2_ii,imm8);       \
      return mask;                                                      \
    }                                                                   \
  }                                                                     \
}


MP_SFS_RELOP_MACRO(==,16) //_CMP_EQ_OS
MP_SFS_RELOP_MACRO(!=,28) //_CMP_NEQ_OS
MP_SFS_RELOP_MACRO(>,14) //_CMP_GT_OS
MP_SFS_RELOP_MACRO(>=,13) //_CMP_GE_OS
MP_SFS_RELOP_MACRO(<,1) //_CMP_LT_OS
MP_SFS_RELOP_MACRO(<=,2) //_CMP_LE_OS

#undef MP_SFS_RELOP_MACRO

#endif

#endif


namespace MaskLogic{

    template<typename T> KOKKOS_INLINE_FUNCTION bool OR(Mask<T> m){
        return (((double) m)!=0.);
    }

    KOKKOS_INLINE_FUNCTION bool OR(bool m){
        return m;
    }

    template<typename T> KOKKOS_INLINE_FUNCTION bool XOR(Mask<T> m){
        return (((double) m)==1./m.getSize());
    }

    KOKKOS_INLINE_FUNCTION bool XOR(bool m){
        return m;
    }

    template<typename T> KOKKOS_INLINE_FUNCTION bool AND(Mask<T> m){
        return (((double) m)==1.);
    }

    KOKKOS_INLINE_FUNCTION bool AND(bool m){
        return m;
    }

}

#endif // STOKHOS_MP_VECTOR_MASKTRAITS_HPP
