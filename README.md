# Trilinos

This is the [Trilinos](https://github.com/trilinos/Trilinos) branch used for [waves](https://github.com/ulgltas/waves).

## How to build Trilinos for waves?

### Debian / Ubuntu

This build is tested on Debian 9.1 and Ubuntu 18.04

```
mkdir build
cd build
../waves/config-gaston.sh 2>&1 | tee cmake.log
grep "Final set of .*enabled SE packages" cmake.log  # (print enabled packages)
make -j 12
make install
```
