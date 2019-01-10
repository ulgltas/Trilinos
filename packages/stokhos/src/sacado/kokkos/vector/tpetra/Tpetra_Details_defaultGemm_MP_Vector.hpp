/*
//@HEADER
// ************************************************************************
//
//          Kokkos: Node API and Parallel Node Kernels
//              Copyright (2008) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
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
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef TPETRA_DETAILS_DEFAULTGEMM_MP_VECTOR_HPP
#define TPETRA_DETAILS_DEFAULTGEMM_MP_VECTOR_HPP

/// \file Tpetra_Details_defaultGemm.hpp
/// \brief Default implementation of local (but process-global) GEMM
///   (dense matrix-matrix multiply), for Tpetra::MultiVector.
///
/// \warning This file, and its contents, are an implementation detail
///   of Tpetra::MultiVector.  Either may disappear or change at any
///   time.

#include "Stokhos_Sacado_Kokkos_MP_Vector.hpp"
#include "Kokkos_Core.hpp"

#include "KokkosBatched_Vector.hpp"


#include "KokkosBatched_Gemm_Decl.hpp"

#include "KokkosBatched_Gemm_Serial_Impl.hpp"
#include <iostream>


#include "Tpetra_Details_defaultGemm.hpp"

namespace Tpetra {
namespace Details {
namespace Blas {
namespace Default {

/*
namespace { // (anonymous)

  template<class ValueType>
  ValueType
  conditionalConjugate (const ValueType& x, const bool conj)
  {
    return conj ? Kokkos::Details::ArithTraits<ValueType>::conj (x) : x;
  }

} // namespace (anonymous)
*/

/// \brief Default implementation of dense matrix-matrix multiply on a
///   single MPI process: <tt>C := alpha*A*B + beta*C</tt>.
///
/// \tparam ViewType1 Type of the first matrix input A.
/// \tparam ViewType2 Type of the second matrix input B.
/// \tparam ViewType3 Type of the third matrix input/output C.
/// \tparam CoefficientType Type of the scalar coefficients alpha and beta.
/// \tparam IndexType Type of the index used in for loops; defaults to \c int.
///
/// ViewType1, ViewType2, and ViewType3 are all Kokkos::View specializations.
template<class ViewType1,
         class ViewType2,
         class ViewType3,
         class Storage,
         class IndexType = int>
void
gemm (const char transA,
      const char transB,
      const Sacado::MP::Vector<Storage>& alpha,
      const ViewType1& A,
      const ViewType2& B,
      const Sacado::MP::Vector<Storage>& beta,
      const ViewType3& C);

} // namespace Default
} // namespace Blas
} // namespace Details
} // namespace Tpetra

#endif // TPETRA_DETAILS_DEFAULTGEMM_MP_VECTOR_HPP
