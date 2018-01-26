#include "TpetraCore_config.h"
#ifdef HAVE_TPETRA_EXPLICIT_INSTANTIATION

#include "Stokhos_Tpetra_ETI_Helpers_MP_Vector.hpp"

#include "Tpetra_Details_defaultGemm_MP_Vector.hpp"
#include "Tpetra_Details_defaultGemm_MP_Vector_def.hpp"

#undef INSTANTIATE_MP_VECTOR_STORAGE


#define INSTANTIATE_MP_VECTOR_STORAGE(INSTMACRO, STORAGE, LO, GO, N)      \
  INSTMACRO( STORAGE, LO, GO, N )

namespace Tpetra {
namespace Details {
namespace Blas {
namespace Default {
/*

  TPETRA_ETI_MANGLING_TYPEDEFS()

  INSTANTIATE_TPETRA_MP_VECTOR_SERIAL(TPETRA_DETAILS_DEFAULTGEMM_MP_VECTOR)
  INSTANTIATE_TPETRA_MP_VECTOR_OPENMP(TPETRA_DETAILS_DEFAULTGEMM_MP_VECTOR)
*/

#define STORAGE Stokhos::StaticFixedStorage<int,double,4,Kokkos::OpenMP>
#define SCALAR Sacado::MP::Vector<STORAGE>
#define VT Kokkos::View<SCALAR**, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>, Kokkos::MemoryTraits<(unsigned int)0>>

template void gemm<VT,VT,VT,STORAGE,int> (const char transA,
      const char transB,
      const SCALAR& alpha,
      const VT& A,
      const VT& B,
      const SCALAR& beta,
      const VT& C);

#undef VT
#undef SCALAR    
#undef STORAGE

#define STORAGE Stokhos::StaticFixedStorage<int,double,8,Kokkos::OpenMP>
#define SCALAR Sacado::MP::Vector<STORAGE>
#define VT Kokkos::View<SCALAR**, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>, Kokkos::MemoryTraits<(unsigned int)0>>

template void gemm<VT,VT,VT,STORAGE,int> (const char transA,
      const char transB,
      const SCALAR& alpha,
      const VT& A,
      const VT& B,
      const SCALAR& beta,
      const VT& C);

#undef VT
#undef SCALAR    
#undef STORAGE

#define STORAGE Stokhos::StaticFixedStorage<int,double,16,Kokkos::OpenMP>
#define SCALAR Sacado::MP::Vector<STORAGE>
#define VT Kokkos::View<SCALAR**, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>, Kokkos::MemoryTraits<(unsigned int)0>>

template void gemm<VT,VT,VT,STORAGE,int> (const char transA,
      const char transB,
      const SCALAR& alpha,
      const VT& A,
      const VT& B,
      const SCALAR& beta,
      const VT& C);

#undef VT
#undef SCALAR    
#undef STORAGE

#define STORAGE Stokhos::StaticFixedStorage<int,double,32,Kokkos::OpenMP>
#define SCALAR Sacado::MP::Vector<STORAGE>
#define VT Kokkos::View<SCALAR**, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>, Kokkos::MemoryTraits<(unsigned int)0>>

template void gemm<VT,VT,VT,STORAGE,int> (const char transA,
      const char transB,
      const SCALAR& alpha,
      const VT& A,
      const VT& B,
      const SCALAR& beta,
      const VT& C);

#undef VT
#undef SCALAR    
#undef STORAGE


#define STORAGE Stokhos::StaticFixedStorage<int,double,4,Kokkos::Serial>
#define SCALAR Sacado::MP::Vector<STORAGE>
#define VT Kokkos::View<SCALAR**, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, Kokkos::MemoryTraits<(unsigned int)0>>

template void gemm<VT,VT,VT,STORAGE,int> (const char transA,
      const char transB,
      const SCALAR& alpha,
      const VT& A,
      const VT& B,
      const SCALAR& beta,
      const VT& C);

#undef VT
#undef SCALAR    
#undef STORAGE

#define STORAGE Stokhos::StaticFixedStorage<int,double,8,Kokkos::Serial>
#define SCALAR Sacado::MP::Vector<STORAGE>
#define VT Kokkos::View<SCALAR**, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, Kokkos::MemoryTraits<(unsigned int)0>>

template void gemm<VT,VT,VT,STORAGE,int> (const char transA,
      const char transB,
      const SCALAR& alpha,
      const VT& A,
      const VT& B,
      const SCALAR& beta,
      const VT& C);

#undef VT
#undef SCALAR    
#undef STORAGE

#define STORAGE Stokhos::StaticFixedStorage<int,double,16,Kokkos::Serial>
#define SCALAR Sacado::MP::Vector<STORAGE>
#define VT Kokkos::View<SCALAR**, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, Kokkos::MemoryTraits<(unsigned int)0>>

template void gemm<VT,VT,VT,STORAGE,int> (const char transA,
      const char transB,
      const SCALAR& alpha,
      const VT& A,
      const VT& B,
      const SCALAR& beta,
      const VT& C);

#undef VT
#undef SCALAR    
#undef STORAGE

#define STORAGE Stokhos::StaticFixedStorage<int,double,32,Kokkos::Serial>
#define SCALAR Sacado::MP::Vector<STORAGE>
#define VT Kokkos::View<SCALAR**, Kokkos::LayoutLeft, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, Kokkos::MemoryTraits<(unsigned int)0>>

template void gemm<VT,VT,VT,STORAGE,int> (const char transA,
      const char transB,
      const SCALAR& alpha,
      const VT& A,
      const VT& B,
      const SCALAR& beta,
      const VT& C);

#undef VT
#undef SCALAR    
#undef STORAGE

} // namespace Default
} // namespace Blas
} // namespace Details
} // namespace Tpetra


#undef INSTANTIATE_MP_VECTOR_STORAGE


#define INSTANTIATE_MP_VECTOR_STORAGE(INSTMACRO, STORAGE, LO, GO, N)      \
  INSTMACRO( Sacado::MP::Vector<STORAGE>, LO, GO, N )

#endif // HAVE_TPETRA_EXPLICIT_INSTANTIATION