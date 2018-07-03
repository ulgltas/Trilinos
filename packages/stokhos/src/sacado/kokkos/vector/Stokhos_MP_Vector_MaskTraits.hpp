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

//#define MASK_STORED_AS_AN_ARRAY

#include "Stokhos_Sacado_Kokkos_MP_Vector.hpp"
#include <iostream>
#include <cmath>
//#include <tuple>
//#include <utility>
#include <initializer_list>

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

template<typename scalar> class Mask;

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
        typedef EnsembleTraits_m<scalar> ET;

#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) = ET::coeff(s,i);

        return *this;
    }
/*
    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator = (const std::pair<scalar,scalar> & KOKKOS_RESTRICT st)
    {
        typedef EnsembleTraits_m<scalar> ET;

#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) = ET::coeff(std::get<0>(st),i);
            else
                ET::coeff(data,i) = ET::coeff(std::get<1>(st),i);

        return *this;
    }
 */

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator = (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        typedef EnsembleTraits_m<scalar> ET;
        auto st_array = st.begin();

#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) = ET::coeff(st_array[0],i);
            else
                ET::coeff(data,i) = ET::coeff(st_array[1],i);

        return *this;
    }


    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator += (const scalar & KOKKOS_RESTRICT s)
    {
        typedef EnsembleTraits_m<scalar> ET;

#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) += ET::coeff(s,i);

        return *this;
    }
/*
    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator += (const std::pair<scalar,scalar> & KOKKOS_RESTRICT st)
    {
        typedef EnsembleTraits_m<scalar> ET;

#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) += ET::coeff(std::get<0>(st),i);
            else
                ET::coeff(data,i) = ET::coeff(std::get<1>(st),i);

        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator += (const std::tuple<scalar,scalar,scalar> & KOKKOS_RESTRICT st)
    {
        typedef EnsembleTraits_m<scalar> ET;

#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) = ET::coeff(std::get<0>(st),i)+ET::coeff(std::get<1>(st),i);
            else
                ET::coeff(data,i) = ET::coeff(std::get<2>(st),i);

        return *this;
    }
*/

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator += (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        typedef EnsembleTraits_m<scalar> ET;
        auto st_array = st.begin();

#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) = ET::coeff(st_array[0],i)+ET::coeff(st_array[1],i);
            else
                ET::coeff(data,i) = ET::coeff(st_array[2],i);

        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator -= (const scalar & KOKKOS_RESTRICT s)
    {
        typedef EnsembleTraits_m<scalar> ET;

#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) -= ET::coeff(s,i);

        return *this;
    }
/*
    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator -= (const std::pair<scalar,scalar> & KOKKOS_RESTRICT st)
    {
        typedef EnsembleTraits_m<scalar> ET;

#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) -= ET::coeff(std::get<0>(st),i);
            else
                ET::coeff(data,i) = ET::coeff(std::get<1>(st),i);

        return *this;
    }
    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator -= (const std::tuple<scalar,scalar,scalar> & KOKKOS_RESTRICT st)
    {
        typedef EnsembleTraits_m<scalar> ET;

#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) = ET::coeff(std::get<0>(st),i)-ET::coeff(std::get<1>(st),i);
            else
                ET::coeff(data,i) = ET::coeff(std::get<2>(st),i);

        return *this;
    }
 */

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator -= (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        typedef EnsembleTraits_m<scalar> ET;
        auto st_array = st.begin();

#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) = ET::coeff(st_array[0],i)-ET::coeff(st_array[1],i);
            else
                ET::coeff(data,i) = ET::coeff(st_array[2],i);

        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator *= (const scalar & KOKKOS_RESTRICT s)
    {
        typedef EnsembleTraits_m<scalar> ET;

#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) *= ET::coeff(s,i);

        return *this;
    }

/*
    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator *= (const std::pair<scalar,scalar> & KOKKOS_RESTRICT st)
    {
        typedef EnsembleTraits_m<scalar> ET;

#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) *= ET::coeff(std::get<0>(st),i);
            else
                ET::coeff(data,i) = ET::coeff(std::get<1>(st),i);

        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator *= (const std::tuple<scalar,scalar,scalar> & KOKKOS_RESTRICT st)
    {
        typedef EnsembleTraits_m<scalar> ET;

#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) = ET::coeff(std::get<0>(st),i)*ET::coeff(std::get<1>(st),i);
            else
                ET::coeff(data,i) = ET::coeff(std::get<2>(st),i);

        return *this;
    }
*/

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator *= (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        typedef EnsembleTraits_m<scalar> ET;
        auto st_array = st.begin();

#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) = ET::coeff(st_array[0],i)*ET::coeff(st_array[1],i);
            else
                ET::coeff(data,i) = ET::coeff(st_array[2],i);

        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator /= (const scalar & KOKKOS_RESTRICT s)
    {
        typedef EnsembleTraits_m<scalar> ET;

#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) /= ET::coeff(s,i);

        return *this;
    }
/*
    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator /= (const std::pair<scalar,scalar> & KOKKOS_RESTRICT st)
    {
        typedef EnsembleTraits_m<scalar> ET;

        #pragma vector aligned
        #pragma ivdep
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) /= ET::coeff(std::get<0>(st),i);
            else
                ET::coeff(data,i) = ET::coeff(std::get<1>(st),i);

        return *this;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator /= (const std::tuple<scalar,scalar,scalar> & KOKKOS_RESTRICT st)
    {
        typedef EnsembleTraits_m<scalar> ET;

#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) = ET::coeff(std::get<0>(st),i)/ET::coeff(std::get<1>(st),i);
            else
                ET::coeff(data,i) = ET::coeff(std::get<2>(st),i);

        return *this;
    }
 */

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) MaskedAssign<scalar>& operator /= (const std::initializer_list<scalar> & KOKKOS_RESTRICT st)
    {
        typedef EnsembleTraits_m<scalar> ET;
        auto st_array = st.begin();

#pragma vector aligned
#pragma ivdep
#pragma unroll
        for(int i=0; i<size; ++i)
            if(m.get(i))
                ET::coeff(data,i) = ET::coeff(st_array[0],i)/ET::coeff(st_array[1],i);
            else
                ET::coeff(data,i) = ET::coeff(st_array[2],i);

        return *this;
    }
};

template<typename scalar> class Mask
{
private:
    static const int size = EnsembleTraits_m<scalar>::size;
#ifdef MASK_STORED_AS_AN_ARRAY
    bool data[size] __attribute__((aligned(64)));
#else
    static const int SIMD_size = 8;
    static const int size_uc = (size == 1 ? 1 : size/SIMD_size);
  
    unsigned char data[size_uc] __attribute__((aligned(64)));
#endif


public:
    Mask(){
        for(int i=0; i<size; ++i)
            this->set(i,false);
    }

    Mask(bool a){
        for(int i=0; i<size; ++i)
            this->set(i,a);
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
        for(int i=0; i<size; ++i)
            m3.set(i,(this->get(i) && m2.get(i)));

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

#ifdef MASK_STORED_AS_AN_ARRAY
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
        int j = i/SIMD_size;
        return (this->data[j] & (1 << i%SIMD_size)) ? true : false;
    }

    KOKKOS_INLINE_FUNCTION __attribute__((always_inline)) void set (int i, bool b)
    {
        int j = i/SIMD_size;
        if(b)
          this->data[j] |= 0x01 << i%SIMD_size;
        else
          this->data[j] &= ~(0x01 << i%SIMD_size);
    }
#endif

    Mask<scalar> operator! ()
    {
        Mask<scalar> m2;
        for(int i=0; i<size; ++i)
            m2.set(i,!(this->get(i)));

        return m2;
    }

    operator bool() const
    {
        return this->data[0];
    }

    operator double() const
    {
        double sum = 0;
        for(int i=0; i<size; ++i)
            sum = sum + this->get(i);

        return sum/size;
    }
};
/*
 template<typename S>  MaskedAssign<Sacado::MP::Vector<S>> operator[] (Sacado::MP::Vector<S> &s, Mask<Sacado::MP::Vector<S>> m) const
 {
 MaskedAssign<Sacado::MP::Vector<S>> maskedAssign = MaskedAssign<Sacado::MP::Vector<S>>(s,m);
 return maskedAssign;
 }
 */

/*
 template<typename S>  MaskedAssign<Sacado::MP::Vector<S>> & operator[] (Sacado::MP::Vector<S> &s, Mask<Sacado::MP::Vector<S>> m)
 {
 MaskedAssign<Sacado::MP::Vector<S>> maskedAssign = MaskedAssign<Sacado::MP::Vector<S>>(s,m);
 return maskedAssign;
 }
 */

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

/*
 template<typename scalar> void mask_assign(Mask<scalar> mask, scalar &dest, scalar if_true, scalar if_false)
 {
 typedef EnsembleTraits_m<scalar> ET;
 for(int i=0; i<ET::size; ++i){
 if (mask[i])
 ET::coeff(dest,i) = ET::coeff(if_true,i);
 else
 ET::coeff(dest,i) = ET::coeff(if_false,i);
 }
 }

 template<typename scalar> void mask_div(Mask<scalar> mask, scalar &dest, scalar if_true, scalar if_true_denominator, scalar if_false)
 {
 typedef EnsembleTraits_m<scalar> ET;
 for(int i=0; i<ET::size; ++i){
 if (mask[i])
 ET::coeff(dest,i) = ET::coeff(if_true,i)/ET::coeff(if_true_denominator,i);
 else
 ET::coeff(dest,i) = ET::coeff(if_false,i);
 }
 }

 */

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


#define MP_VECTOR_RELOP_MACRO(OP)                                       \
namespace Sacado {                                                      \
  namespace MP {                                                        \
                                                                        \
    template <typename S>                                               \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask<Vector<S> >                                                    \
    operator OP (const Vector<S> &a1,                                   \
                 const Vector<S> &a2)                                   \
    {                                                                   \
      typedef EnsembleTraits_m<Vector<S>> ET;                           \
      Mask<Vector<S> > mask;                                            \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      for(int i=0; i<ET::size; ++i)                                     \
        mask.set(i, ET::coeff(a1,i) OP ET::coeff(a2,i));                \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename S>                                               \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask<Vector<S> >                                                    \
    operator OP (const Vector<S> &a1,                                   \
                 const typename S::value_type &a2)                      \
    {                                                                   \
      typedef EnsembleTraits_m<Vector<S>> ET;                           \
      Mask<Vector<S> > mask;                                            \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      for(int i=0; i<ET::size; ++i)                                     \
        mask.set(i, ET::coeff(a1,i) OP a2);                             \
      return mask;                                                      \
    }                                                                   \
                                                                        \
    template <typename S>                                               \
    KOKKOS_INLINE_FUNCTION                                              \
    Mask<Vector<S> >                                                    \
    operator OP (const typename S::value_type &a1,                      \
                 const Vector<S> &a2)                                   \
    {                                                                   \
      typedef EnsembleTraits_m<Vector<S>> ET;                           \
      Mask<Vector<S> > mask;                                            \
      _Pragma("vector aligned")                                         \
      _Pragma("ivdep")                                                  \
      _Pragma("unroll")                                                 \
      for(int i=0; i<ET::size; ++i)                                     \
        mask.set(i, a1 OP ET::coeff(a2,i));                             \
      return mask;                                                      \
    }                                                                   \
  }                                                                     \
}

MP_VECTOR_RELOP_MACRO(==)
MP_VECTOR_RELOP_MACRO(!=)
MP_VECTOR_RELOP_MACRO(>)
MP_VECTOR_RELOP_MACRO(>=)
MP_VECTOR_RELOP_MACRO(<)
MP_VECTOR_RELOP_MACRO(<=)
MP_VECTOR_RELOP_MACRO(<<=)
MP_VECTOR_RELOP_MACRO(>>=)
MP_VECTOR_RELOP_MACRO(&)
MP_VECTOR_RELOP_MACRO(|)

#undef MP_VECTOR_RELOP_MACRO

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
      for(int i=0; i<v1.size(); ++i)                                    \
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
