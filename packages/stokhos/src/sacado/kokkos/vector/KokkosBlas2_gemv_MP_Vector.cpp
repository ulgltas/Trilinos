#include "KokkosBlas2_gemv_MP_Vector.hpp"
#include "KokkosBlas2_gemv_MP_Vector_def.hpp"

#define ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_VA(VA,VX,VY,SCALAR)                \
template void KokkosBlas::gemv<VT,VT,VT,STORAGE,int> (const char trans[], \
      const SCALAR& alpha,                                                \
      const VA& A,                                                        \
      const VX& x,                                                        \
      const SCALAR& beta,                                                 \
      const VY& y);

#define ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_ENSEMBLE_SIZE_OpenMP(s)             \
  typedef Stokhos::StaticFixedStorage<int,double,s,Kokkos::OpenMP> STORAGE;\
  typedef Sacado::MP::Vector<STORAGE> SCALAR;                              \
  typedef Kokkos::View<SCALAR**, Kokkos::LayoutLeft,                       \
          Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>,               \
          Kokkos::MemoryTraits<(unsigned int)0>> VA;                       \
  typedef Kokkos::View<SCALAR*, Kokkos::LayoutLeft,                        \
          Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>,               \
          Kokkos::MemoryTraits<(unsigned int)0>> VX;                       \
  typedef Kokkos::View<SCALAR*, Kokkos::LayoutLeft,                        \
          Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>,               \
          Kokkos::MemoryTraits<(unsigned int)0>> VY;                       \
  ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_VA(VA,VX,VY,SCALAR)                       \
  typedef Kokkos::View<SCALAR**, Kokkos::LayoutRight,                      \
          Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>,               \
          Kokkos::MemoryTraits<(unsigned int)0>> VA;                       \
  typedef Kokkos::View<SCALAR*, Kokkos::LayoutRight,                       \
          Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>,               \
          Kokkos::MemoryTraits<(unsigned int)0>> VX;                       \
  typedef Kokkos::View<SCALAR*, Kokkos::LayoutRight,                       \
          Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>,               \
          Kokkos::MemoryTraits<(unsigned int)0>> VY;                       \
  ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_VA(VA,VX,VY,SCALAR)
  
ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_ENSEMBLE_SIZE_OpenMP(8)
ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_ENSEMBLE_SIZE_OpenMP(16)
ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_ENSEMBLE_SIZE_OpenMP(24)
ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_ENSEMBLE_SIZE_OpenMP(32)
