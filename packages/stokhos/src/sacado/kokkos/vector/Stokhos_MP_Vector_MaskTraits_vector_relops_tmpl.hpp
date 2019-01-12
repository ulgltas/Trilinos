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

#ifndef STOKHOS_MP_VECTOR_MASK_USE_II
namespace Sacado {
  namespace MP {
    template <typename S>
    KOKKOS_INLINE_FUNCTION __attribute__((always_inline))
    Mask<Vector<S> >
    operator OPNAME (const Vector<S> &a1,
                 const Vector<S> &a2)
    {
      typedef EnsembleTraits_m<Vector<S>> ET;
      Mask<Vector<S> > mask;
#ifdef STOKHOS_HAVE_PRAGMA_IVDEP
#pragma ivdep
#endif
#ifdef STOKHOS_HAVE_PRAGMA_VECTOR_ALIGNED
#pragma vector aligned
#endif
#ifdef STOKHOS_HAVE_PRAGMA_UNROLL
#pragma unroll
#endif
      for(int i=0; i<ET::size; ++i)
        mask.set(i, ET::coeff(a1,i) OPNAME ET::coeff(a2,i));
      return mask;
    }

    template <typename S>
    KOKKOS_INLINE_FUNCTION __attribute__((always_inline))
    Mask<Vector<S> >
    operator OPNAME (const Vector<S> &a1,
                 const typename S::value_type &a2)
    {
      typedef EnsembleTraits_m<Vector<S>> ET;
      Mask<Vector<S> > mask;
#ifdef STOKHOS_HAVE_PRAGMA_IVDEP
#pragma ivdep
#endif
#ifdef STOKHOS_HAVE_PRAGMA_VECTOR_ALIGNED
#pragma vector aligned
#endif
#ifdef STOKHOS_HAVE_PRAGMA_UNROLL
#pragma unroll
#endif
      for(int i=0; i<ET::size; ++i)
        mask.set(i, ET::coeff(a1,i) OPNAME a2);
      return mask;
    }

    template <typename S>
    KOKKOS_INLINE_FUNCTION __attribute__((always_inline))
    Mask<Vector<S> >
    operator OPNAME (const typename S::value_type &a1,
                 const Vector<S> &a2)
    {
      typedef EnsembleTraits_m<Vector<S>> ET;
      Mask<Vector<S> > mask;
#ifdef STOKHOS_HAVE_PRAGMA_IVDEP
#pragma ivdep
#endif
#ifdef STOKHOS_HAVE_PRAGMA_VECTOR_ALIGNED
#pragma vector aligned
#endif
#ifdef STOKHOS_HAVE_PRAGMA_UNROLL
#pragma unroll
#endif
      for(int i=0; i<ET::size; ++i)
        mask.set(i, a1 OPNAME ET::coeff(a2,i));
      return mask;
    }
  }
}

#else
namespace Sacado {
  namespace MP {
    template <typename S>
    KOKKOS_INLINE_FUNCTION __attribute__((always_inline))
    Mask<Vector<S> >
    operator OPNAME (const Vector<S> &a1,
                 const Vector<S> &a2)
    {
      Mask<Vector<S> > mask;
#ifdef STOKHOS_HAVE_PRAGMA_UNROLL
#pragma unroll
#endif
      for(int i=0; i<mask.size_uc; ++i)
        mask.data[i] = _mm512_cmp_pd_mask(M512D_ENSEMBLE_LOAD(a1,i),M512D_ENSEMBLE_LOAD(a2,i),imm8);
      return mask;
    }
    template <typename S>
    KOKKOS_INLINE_FUNCTION __attribute__((always_inline))
    Mask<Vector<S> >
    operator OPNAME (const Vector<S> &a1,
                 const double &a2)
    {
      Mask<Vector<S> > mask;
      __m512d a2_ii = _mm512_set1_pd(a2);
#ifdef STOKHOS_HAVE_PRAGMA_UNROLL
#pragma unroll
#endif
      for(int i=0; i<mask.size_uc; ++i)
        mask.data[i] = _mm512_cmp_pd_mask(M512D_ENSEMBLE_LOAD(a1,i),a2_ii,imm8);
      return mask;
    }
    
    template <typename S>
    KOKKOS_INLINE_FUNCTION __attribute__((always_inline))
    Mask<Vector<S> >
    operator OPNAME (const typename S::value_type &a1,
                 const Vector<S> &a2)
    {
      Mask<Vector<S> > mask;
      __m512d a1_ii = _mm512_set1_pd(a1);
#ifdef STOKHOS_HAVE_PRAGMA_UNROLL
#pragma unroll
#endif
      for(int i=0; i<mask.size_uc; ++i)
        mask.data[i] = _mm512_cmp_pd_mask(a1_ii,M512D_ENSEMBLE_LOAD(a2,i),imm8);
      return mask;
    }
  }
}
#endif
