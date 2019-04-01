#!/bin/bash

rm -f CMakeCache.txt;
rm -rf CMakeFiles

EXTRA_ARGS=$@

cmake \
 -D CMAKE_INSTALL_PREFIX:PATH="~/local/trilinos" \
 -D CMAKE_BUILD_TYPE:STRING=RELEASE \
 -D Trilinos_ENABLE_INSTALL_CMAKE_CONFIG_FILES:BOOL=ON \
 -D Trilinos_ENABLE_DEBUG:BOOL=OFF \
 -D Trilinos_ENABLE_CHECKED_STL:BOOL=OFF \
 -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
 -D Tpetra_INST_SERIAL=ON \
 -D Tpetra_INST_OPENMP=ON \
 -D Tpetra_INST_PTHREAD=OFF \
 -D Tpetra_INST_FLOAT=OFF \
 -D Tpetra_INST_COMPLEX_FLOAT=OFF \
 -D Tpetra_INST_COMPLEX_DOUBLE=OFF \
 -D Tpetra_INST_INT_LONG=OFF \
 -D Tpetra_INST_INT_LONG_LONG=OFF \
 -D Tpetra_INST_INT_UNSIGNED=OFF \
 -D Trilinos_WARNINGS_AS_ERRORS_FLAGS:STRING="" \
 -D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
 -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=OFF \
 -D Trilinos_ENABLE_OpenMP:BOOL=ON \
 -D Trilinos_ENABLE_CXX11:BOOL=ON \
 -D BUILD_SHARED_LIBS:BOOL=ON \
 -D DART_TESTING_TIMEOUT:STRING=600 \
 -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
 -D Trilinos_ENABLE_Fortran=OFF \
 -D TPL_ENABLE_MPI:BOOL=ON \
 -D TPL_ENABLE_BinUtils=OFF \
 -D TPL_ENABLE_Pthread=OFF \
 -D Kokkos_ENABLE_Serial=ON \
 -D Trilinos_ENABLE_Moertel:BOOL=ON \
 -D Trilinos_ENABLE_MueLu:BOOL=ON \
 -D Trilinos_ENABLE_Amesos2:BOOL=ON \
 -D Trilinos_ENABLE_PyTrilinos:BOOL=ON \
 -D Trilinos_ENABLE_Stokhos:BOOL=ON \
 -D Trilinos_ENABLE_STK:BOOL=OFF \
 -D Trilinos_ENABLE_SEACAS=OFF \
 -D Trilinos_ENABLE_Sacado:BOOL=ON \
 -D Trilinos_ENABLE_Ifpack2:BOOL=ON \
 -D Trilinos_ENABLE_Thyra:BOOL=ON \
 -D Trilinos_ENABLE_Teko:BOOL=ON \
 -D Stokhos_ENABLE_EXAMPLES:BOOL=OFF \
 -D Stokhos_ENABLE_TESTS:BOOL=OFF \
 -D Stokhos_ENABLE_Ensemble_Reduction=OFF \
 -D Amesos2_ENABLE_Basker=ON \
 -D Amesos2_ENABLE_KLU2=ON \
 -D TPL_ENABLE_ExodusII:BOOL=OFF \
 -D TPL_ENABLE_Matio:BOOL=OFF \
 -D TPL_ENABLE_Nemesis:BOOL=OFF \
 -D Stokhos_ENABLE_TESTS:BOOL=ON \
 -D MueLu_ENABLE_TESTS:BOOL=OFF \
 -D PyTrilinos_DOCSTRINGS:BOOL=OFF \
 -D PyTrilinos_ENABLE_Tpetra:BOOL=OFF \
 -D TPL_ENABLE_MKL:BOOL=ON \
 -D MKL_LIBRARY_DIRS="/opt/intel/mkl/lib/intel64" \
 -D MKL_INCLUDE_DIRS="/opt/intel/mkl/include/" \
 -D TpetraCore_Threaded_MKL:BOOL=ON \
 -D MPI_BASE_DIR:PATH="/usr/lib/x86_64-linux-gnu/openmpi/" \
$EXTRA_ARGS \
 ..

