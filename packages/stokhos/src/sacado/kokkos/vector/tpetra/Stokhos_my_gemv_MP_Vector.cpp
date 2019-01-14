#include "Stokhos_my_gemv_MP_Vector_decl.hpp"
#include "Stokhos_my_gemv_MP_Vector_def.hpp"

#define ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_SLDM(SCALAR,LAYOUT,DEVICE,MEMORY)                 \
template void my_gemv                                                                    \
  <SCALAR,                                                                               \
  Kokkos::View<SCALAR**,LAYOUT,DEVICE,MEMORY>,                                           \
  Kokkos::View<SCALAR*,LAYOUT,DEVICE,MEMORY>,                                            \
  Kokkos::View<SCALAR*,LAYOUT,DEVICE,MEMORY>>                                            \
  (const char trans[],                                                                   \
  const typename Kokkos::View<SCALAR**,LAYOUT,DEVICE,MEMORY>::const_value_type& alpha,   \
  const Kokkos::View<SCALAR**,LAYOUT,DEVICE,MEMORY>& A,                                  \
  const Kokkos::View<SCALAR*,LAYOUT,DEVICE,MEMORY>& x,                                   \
  const typename Kokkos::View<SCALAR*,LAYOUT,DEVICE,MEMORY>::const_value_type& beta,     \
  const Kokkos::View<SCALAR*,LAYOUT,DEVICE,MEMORY>& y);

#define ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_ENSEMBLE_SIZE_OpenMP(S)             \
  typedef Stokhos::StaticFixedStorage<int,double,S,Kokkos::OpenMP>         \
  STORAGE_OPENMP_ ## S;                                                    \
  typedef Sacado::MP::Vector<STORAGE_OPENMP_ ## S> SCALAR_OPENMP_ ## S;    \
  typedef Kokkos::LayoutLeft LAYOUT_ ## S;                                 \
  typedef Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace> DEVICE_ ## S;  \
  typedef Kokkos::MemoryTraits<(unsigned int)0> MT_ ## S;                  \
  ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_SLDM(SCALAR_OPENMP_ ## S,                 \
                                      LAYOUT_ ## S,                        \
                                      DEVICE_ ## S,                        \
                                      MT_ ## S)

ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_ENSEMBLE_SIZE_OpenMP(4)
ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_ENSEMBLE_SIZE_OpenMP(8)
ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_ENSEMBLE_SIZE_OpenMP(16)
ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_ENSEMBLE_SIZE_OpenMP(24)
ETI_KOKKOSBLAS2_GEMV_MP_VECTOR_ENSEMBLE_SIZE_OpenMP(32)
