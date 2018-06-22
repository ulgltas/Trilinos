// @HEADER
// ***********************************************************************
//
//                    Teuchos: Common Tools Package
//                 Copyright (2004) Sandia Corporation
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
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ***********************************************************************
// @HEADER

#ifndef _TEUCHOS_BLAS_MP_VECTOR_HPP_
#define _TEUCHOS_BLAS_MP_VECTOR_HPP_

#include "Teuchos_BLAS.hpp"
//#include "Sacado_MP_Vector.hpp"

#include "Stokhos_MP_Vector_MaskTraits.hpp"

// Specialize some things used in the default BLAS implementation that
// don't seem correct for MP::Vector scalar type
namespace Teuchos {
    
    namespace details {
        
        template<typename Storage>
        class GivensRotator<Sacado::MP::Vector<Storage>, false> {
        public:
            typedef Sacado::MP::Vector<Storage> ScalarType;
            typedef ScalarType c_type;
            
            void
            ROTG (ScalarType* da,
                  ScalarType* db,
                  ScalarType* c,
                  ScalarType* s) const {
                typedef ScalarTraits<ScalarType> STS;
                
                ScalarType r, roe, scale, z, da_scaled, db_scaled;
                auto m_da = (STS::magnitude (*da) > STS::magnitude (*db));
                mask_assign(m_da,roe) = {*da,*db};
                
                scale = STS::magnitude (*da) + STS::magnitude (*db);
                
                auto m_scale = scale != STS::zero();
                
                da_scaled = *da;
                db_scaled = *db;
                
                *c = *da;
                *s = *db;
                
                ScalarType tmp = STS::one();
                mask_assign(m_scale,tmp) /= scale;
                
                mask_assign(m_scale,da_scaled) *= tmp;
                mask_assign(m_scale,db_scaled) *= tmp;
                
                r = scale * STS::squareroot (da_scaled*da_scaled + db_scaled*db_scaled);
                auto m_roe = roe < 0;
                mask_assign(m_roe,r) = -r;
                
                tmp = STS::one();
                mask_assign(m_scale,tmp) /= r;
                
                mask_assign(m_scale,*c) *= tmp;
                mask_assign(m_scale,*s) *= tmp;
                
                mask_assign(!m_scale,*c) = STS::one();
                mask_assign(!m_scale,*s) = STS::zero();
                
                
                mask_assign(*c != STS::zero(),z) /= {STS::one(),*c,STS::zero()};
                mask_assign(!m_scale,z) = STS::zero();
                mask_assign(m_da,z) = *s;
                
                *da = r;
                *db = z;
            }
        };
    } // namespace details
    
} // namespace Teuchos

#endif // _TEUCHOS_BLAS__MP_VECTOR_HPP_
