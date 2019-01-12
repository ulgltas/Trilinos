#include "KokkosBlas2_gemv_MP_Vector.hpp"
#include "KokkosBlas2_gemv_MP_Vector_def.hpp"

#define ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_VA(VA,VX,VY,SCALAR)                 \
template void KokkosBlas::gemv<VA,VX,VY> (const char trans[],              \
      const SCALAR& alpha,                                                 \
      const VA& A,                                                         \
      const VX& x,                                                         \
      const SCALAR& beta,                                                  \
      const VY& y);

#define ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_ENSEMBLE_SIZE_OpenMP(S)             \
  typedef Stokhos::StaticFixedStorage<int,double,S,Kokkos::OpenMP>         \
  STORAGE_OPENMP_ ## S;                                                    \
  typedef Sacado::MP::Vector<STORAGE_OPENMP_ ## S> SCALAR_OPENMP_ ## S;    \
  typedef Kokkos::View<SCALAR_OPENMP_ ## S**, Kokkos::LayoutLeft,          \
          Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>,               \
          Kokkos::MemoryTraits<(unsigned int)0>> VA_LEFT_OPENMP_ ## S;     \
  typedef Kokkos::View<SCALAR_OPENMP_ ## S*, Kokkos::LayoutLeft,           \
          Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>,               \
          Kokkos::MemoryTraits<(unsigned int)0>> VX_LEFT_OPENMP_ ## S;     \
  typedef Kokkos::View<SCALAR_OPENMP_ ## S*, Kokkos::LayoutLeft,           \
          Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>,               \
          Kokkos::MemoryTraits<(unsigned int)0>> VY_LEFT_OPENMP_ ## S;     \
  ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_VA(VA_LEFT_OPENMP_ ## S,                  \
                                    VX_LEFT_OPENMP_ ## S,                  \
                                    VY_LEFT_OPENMP_ ## S,                  \
                                    SCALAR_OPENMP_ ## S)                   \
  typedef Kokkos::View<SCALAR_OPENMP_ ## S**, Kokkos::LayoutRight,         \
          Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>,               \
          Kokkos::MemoryTraits<(unsigned int)0>> VA_RIGHT_OPENMP_ ## S;    \
  typedef Kokkos::View<SCALAR_OPENMP_ ## S*, Kokkos::LayoutRight,          \
          Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>,               \
          Kokkos::MemoryTraits<(unsigned int)0>> VX_RIGHT_OPENMP_ ## S;    \
  typedef Kokkos::View<SCALAR_OPENMP_ ## S*, Kokkos::LayoutRight,          \
          Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>,               \
          Kokkos::MemoryTraits<(unsigned int)0>> VY_RIGHT_OPENMP_ ## S;    \
  ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_VA(VA_RIGHT_OPENMP_ ## S,                 \
                                    VX_RIGHT_OPENMP_ ## S,                 \
                                    VY_RIGHT_OPENMP_ ## S,                 \
                                    SCALAR_OPENMP_ ## S)
  
ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_ENSEMBLE_SIZE_OpenMP(8)
ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_ENSEMBLE_SIZE_OpenMP(16)
ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_ENSEMBLE_SIZE_OpenMP(24)
ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_ENSEMBLE_SIZE_OpenMP(32)
