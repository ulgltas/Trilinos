/*
// @HEADER
// ************************************************************************
//             FEI: Finite Element Interface to Linear Solvers
//                  Copyright (2005) Sandia Corporation.
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation, the
// U.S. Government retains certain rights in this software.
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
// Questions? Contact Alan Williams (william@sandia.gov) 
//
// ************************************************************************
// @HEADER
*/


#ifndef _fei_Factory_Trilinos_hpp_
#define _fei_Factory_Trilinos_hpp_

#include "fei_trilinos_macros.hpp"

#include <fei_mpi.h>

#include <fei_Include_Trilinos.hpp>

#ifdef HAVE_FEI_EPETRA
#include <fei_VectorTraits_Epetra.hpp>
#include <fei_MatrixTraits_Epetra.hpp>
#include <fei_Trilinos_Helpers.hpp>
#include <fei_LinProbMgr_EpetraBasic.hpp>
#endif

#include <fei_Factory.hpp>
#include <fei_ParameterSet.hpp>
#include <fei_Reducer.hpp>
#include <fei_Vector_Impl.hpp>
#include <fei_Matrix_Impl.hpp>
#include <fei_MatrixGraph_Impl2.hpp>
#include <fei_SparseRowGraph.hpp>
#include <fei_utils.hpp>

#undef fei_file
#define fei_file "fei_Factory_Trilinos.hpp"
#include <fei_ErrMacros.hpp>

/*** Implementation of an fei::Factory which creates instances that use Trilinos
     objects (Epetra and AztecOO) as the underlying objects.
*/
class Factory_Trilinos : public fei::Factory {
 public:
  Factory_Trilinos(MPI_Comm comm);

  virtual ~Factory_Trilinos();

  /** Implementation of fei::Factory::clone() */
  fei::SharedPtr<fei::Factory> clone() const
    {
      fei::SharedPtr<fei::Factory> factory(new Factory_Trilinos(comm_));
      return(factory);
    }

    /** Implementation of fei::Factory::parameters() */
    virtual int parameters(int numParams,
                           const char* const* paramStrings);

    /** Implementation of fei::Factory::parameters() */
    virtual void parameters(const fei::ParameterSet& parameterset);

  /** Implementation of fei::MatrixGraph::Factory::createMatrixGraph() */
  fei::SharedPtr<fei::MatrixGraph>
    createMatrixGraph(fei::SharedPtr<fei::VectorSpace> rowSpace,
                      fei::SharedPtr<fei::VectorSpace> colSpace,
                      const char* name);

  /** Implementation of fei::Vector::Factory::createVector() */
  fei::SharedPtr<fei::Vector>
    createVector(fei::SharedPtr<fei::VectorSpace> vecSpace, int numVectors=1);

#ifdef HAVE_FEI_EPETRA
  /** Wrap fei::Vector around existing Epetra_MultiVector.
      If the specified vector-space isn't compatible with the multi-vector's size,
      then return a null fei::Vector.
  */
  fei::SharedPtr<fei::Vector>
    wrapVector(fei::SharedPtr<fei::VectorSpace> vecSpace,
               fei::SharedPtr<Epetra_MultiVector> multiVec);

  /** Wrap fei::Vector around existing Epetra_MultiVector.
      If the specified matrix-graph's vector-space isn't compatible with
      the multi-vector's size, then return a null fei::Vector.
  */
  fei::SharedPtr<fei::Vector>
    wrapVector(fei::SharedPtr<fei::MatrixGraph> matGraph,
               fei::SharedPtr<Epetra_MultiVector> multiVec);
#endif

  /** Implementation of fei::Vector::Factory::createVector() */
  fei::SharedPtr<fei::Vector>
    createVector(fei::SharedPtr<fei::VectorSpace> vecSpace,
		  bool isSolutionVector,
		  int numVectors=1);

  /** Produce an instance of a Vector using a MatrixGraph. */
  fei::SharedPtr<fei::Vector>
    createVector(fei::SharedPtr<fei::MatrixGraph> matrixGraph,
		  int numVectors=1);

  /** Produce an instance of a Vector using a MatrixGraph. */
  fei::SharedPtr<fei::Vector>
    createVector(fei::SharedPtr<fei::MatrixGraph> matrixGraph,
		  bool isSolutionVector,
		  int numVectors=1);

  fei::SharedPtr<fei::Matrix>
    createMatrix(fei::SharedPtr<fei::MatrixGraph> matrixGraph);

  fei::SharedPtr<fei::Solver> createSolver(const char* name=0);

  int getOutputLevel() const { return(outputLevel_); }

 private:
  void create_LinProbMgr(bool replace_if_already_created=false);

  MPI_Comm comm_;

  fei::SharedPtr<fei::Reducer> reducer_;
  fei::SharedPtr<fei::LinearProblemManager> lpm_epetrabasic_;
  bool use_lpm_epetrabasic_;
  bool useAmesos_;
  bool useBelos_;
  bool use_feiMatrixLocal_;
  bool blockEntryMatrix_;
  bool orderRowsWithLocalColsFirst_;

  int outputLevel_;
};

#endif // _Factory_Trilinos_hpp_

