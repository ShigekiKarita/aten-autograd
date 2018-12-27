# ATNN

computational graph library for [ATEN](https://github.com/zdevito/ATen)

## TODO

+ computational graph
    + support broadcast in Variable (currently operator+ only)
    + retain_grad=true option like pytorch
    + dot export
+ gradient check
    + test_rnn.cpp
+ optimizers
    + gradient null check
    + support double backward?
+ examples
    + PTB for RNNLM
    + TIMIT for encoder-decoder attention
+ CUDNN support
    + already supported in ATen? https://github.com/zdevito/ATen/commit/6d8547785755d279a7b3457e8b6ba666e206ef86
    + use pytorch functions https://github.com/pytorch/pytorch/blob/master/torch/csrc/cudnn/Conv.h


## Concepts

| ATNN     | PyTorch   |
| :----:   | :-------: |
| Tensor   | Tensor    |
| Variable | Variable  |
| Function | Function  |
| Chain    | Module    |

+ note: I avoid `module` because it will be C++ (Module-TS) keyword.

## Documentation

how to build `cd doc; doxygen` and see `doc/doxgen/html/index.html`

## Requirements

tested eviroment

+ ATEN (see submodule commit-id and `test/Makefile`)
+ g++ 6.3.0, 7.2.0 (4.9.4, 5.4.1 failed)
+ clang++ 4.0.0, 4.0.1

note: anaconda's libgcc causes some symbol undefined errors in libstdc++


## Usage

see .cpp rule of test/Makefile


## Test

``` bash
cd test

# this rule also builds ATen submodule into test/build/stage at first
make test

# with Boost.StackTrace in 1.65.0 for better error messages
make USE_BOOST=true test

# with compiler specification
make CXX=clang++-4.0 test

# mnist example
make test-mnist

# coverage report
make coverage
xdg-open ./html-gcovr/coverage.html
```

if you enable Boost.Stacktrace, errors can be printed as follows:
``` console
$ make USE_BOOST=true test_rnn.out
g++ -o test_rnn.out test_rnn.cpp -std=gnu++14 -coverage -fno-inline -fno-inline-small-functions -fno-default-inline -g3 -O0 -ftrapv -Wall -Wextra -Wno-unused-function -D_LIBCPP_DEBUG -D_GLIBCXX_DEBUG   -DBOOST_ENABLE_ASSERT_DEBUG_HANDLER -DHAVE_BOOST -DBOOST_STACKTRACE_USE_ADDR2LINE -Ibuild/stage/include -I.  -lATen -ldl -lboost_stacktrace_addr2line
$ ./test_rnn.out
./test_rnn.out===== ATNN-Assetion failed =====
Expression: (gy.size()) == (1)
Function:   atnn::TList atnn::function::Tanh::impl_backward(const TList&) in ./atnn/function.hpp(469)
Message:    0 != 1

=========== Backtrace ==========
 0# _ZN5boost10stacktrace16basic_stacktraceISaINS0_5frameEEEC4Ev at /home/skarita/tool/boost_1_65_1/stage/default/include/boost/stacktrace/stacktrace.hpp:130 (discriminator 4)
 1# atnn::function::Tanh::impl_backward(std::__debug::vector<at::Tensor, std::allocator<at::Tensor> > const&) at /home/skarita/Documents/repos/atnn/test/./atnn/function.hpp:469
 2# atnn::function::Function<atnn::function::Tanh>::backward(std::__debug::vector<at::Tensor, std::allocator<at::Tensor> >) at /home/skarita/Documents/repos/atnn/test/./atnn/function.hpp:145
 3# atnn::function::Function<atnn::function::Tanh>::compute_grad_inputs() at /home/skarita/Documents/repos/atnn/test/./atnn/function.hpp:60 (discriminator 3)
 4# atnn::Variable::backward(at::Tensor) at /home/skarita/Documents/repos/atnn/test/./atnn/variable.hpp:156 (discriminator 2)
 5# atnn::Variable::backward(at::Tensor) at /home/skarita/Documents/repos/atnn/test/./atnn/variable.hpp:157 (discriminator 3)
 6# atnn::Variable::backward(at::Tensor) at /home/skarita/Documents/repos/atnn/test/./atnn/variable.hpp:157 (discriminator 3)
 7# atnn::Variable::backward(at::Tensor) at /home/skarita/Documents/repos/atnn/test/./atnn/variable.hpp:157 (discriminator 3)
 8# atnn::Variable::backward(at::Tensor) at /home/skarita/Documents/repos/atnn/test/./atnn/variable.hpp:157 (discriminator 3)
 9# _ZZ4mainENKUlT_E_clIPFRN2at4TypeENS2_10ScalarTypeEEEEDaS_ at /home/skarita/Documents/repos/atnn/test/test_rnn.cpp:48 (discriminator 3)
10# _ZN4atnn11test_commonIZ4mainEUlT_E_EEviPPcS1_b at /home/skarita/Documents/repos/atnn/test/./atnn/testing.hpp:130
11# main at /home/skarita/Documents/repos/atnn/test/test_rnn.cpp:40
12# __libc_start_main in /lib64/libc.so.6
13# _start in ./test_rnn.out

[1]    30918 abort (core dumped)  ./test_rnn.out
```

