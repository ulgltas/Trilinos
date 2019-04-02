# Trilinos for waves

This is the [Trilinos](https://github.com/trilinos/Trilinos) branch used for [waves](https://github.com/ulgltas/waves).

This branch is based on [Kim Liegeois's fork of Trilinos](https://github.com/kliegeois/Trilinos)

## How to build Trilinos for waves?

### Debian / Ubuntu (gcc)

This build is tested on Debian 9.1 and Ubuntu 18.04

```
mkdir build
cd build
../waves/config-gaston.sh 2>&1 | tee cmake.log
grep "Final set of .*enabled SE packages" cmake.log  # (print enabled packages - se below)
make -j 12
make install
```

## Info: Enabled SE packages

Final set of enabled SE packages:  KokkosCore KokkosContainers KokkosAlgorithms Kokkos TeuchosCore TeuchosParser TeuchosParameterList TeuchosComm TeuchosNumerics TeuchosRemainder TeuchosKokkosCompat TeuchosKokkosComm Teuchos KokkosKernels RTOp Sacado Epetra Triutils EpetraExt TpetraClassic TpetraCore Tpetra TrilinosSS ThyraCore ThyraEpetraAdapters ThyraEpetraExtAdapters ThyraTpetraAdapters Thyra Xpetra AztecOO Galeri Amesos Ifpack ML Belos Amesos2 Anasazi Ifpack2 Stratimikos Teko Moertel MueLu Stokhos PyTrilinos 44

Final set of non-enabled SE packages:  TrilinosFrameworkTests Gtest KokkosExample MiniTensor Zoltan Shards GlobiPack TpetraTSQR Domi OptiPack Isorropia Pliris Claps Pamgen Zoltan2 ShyLU_NodeHTS ShyLU_NodeTacho ShyLU_NodeBasker ShyLU_NodeFastILU ShyLU_Node SEACASExodus SEACASExodus_for SEACASExoIIv2for32 SEACASNemesis SEACASIoss SEACASChaco SEACASAprepro_lib SEACASSupes SEACASSuplib SEACASSuplibC SEACASSuplibCpp SEACASSVDI SEACASPLT SEACASAlgebra SEACASAprepro SEACASBlot SEACASConjoin SEACASEjoin SEACASEpu SEACASExo2mat SEACASExodiff SEACASExomatlab SEACASExotxt SEACASExo_format SEACASEx1ex2v2 SEACASExotec2 SEACASFastq SEACASGjoin SEACASGen3D SEACASGenshell SEACASGrepos SEACASExplore SEACASMapvarlib SEACASMapvar SEACASMapvar-kd SEACASMat2exo SEACASNemslice SEACASNemspread SEACASNumbers SEACASSlice SEACASTxtexo SEACASEx2ex1v2 SEACAS Trioscommsplitter Triossupport Triosnnti Triosnssi Triosprograms Triosexamples Triostests Triosnetcdf-service Trios Komplex FEI TriKota Intrepid Intrepid2 STKUtil STKSimd STKTopology STKMesh STKNGP STKIO STKNGP_TEST STKUnit_test_utils STKMath STKSearch STKSearchUtil STKTransfer STKTools STKUnit_tests STKDoc_tests STKExp STKExprEval STK Phalanx NOX ShyLU_DDBDDC ShyLU_DDFROSch ShyLU_DDCore ShyLU_DDCommon ShyLU_DD ShyLU Rythmos Tempus ROL Piro PanzerCore PanzerDofMgr PanzerDiscFE PanzerAdaptersSTK PanzerAdaptersIOSS PanzerMiniEM PanzerExprEval Panzer NewPackage TrilinosCouplings PikeBlackBox PikeImplicit Pike 120
