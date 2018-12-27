CXX := g++

ATEN_ROOT := build/stage
INCPATH := -I$(ATEN_ROOT)/include -I. -I$(CONDA_PREFIX)/include
LIBPATH := -L$(ATEN_ROOT)/lib -L$(CONDA_PREFIX)/lib
LIBS := -lATen
H5_LIB := -lhdf5 -lhdf5_cpp
RUNTIME_LIB := $(ATEN_ROOT)/lib:$(CONDA_PREFIX)/lib
KALDI_ROOT := kaldi
KALDI_FLAGS := -DHAVE_MKL -lkaldi-base -lkaldi-util -lkaldi-matrix \
 -isystem$(KALDI_ROOT)/src -I$(KALDI_ROOT)/src/util -isystem$(KALDI_ROOT)/src/matrix -isystem$(KALDI_ROOT)/tools/openfst/include \
 -L$(KALDI_ROOT)/src/lib -L$(KALDI_ROOT)/tools/openfst/lib -pthread

RAPIDJSON_VER=1.1.0

ifdef RELEASE
CXX_FLAGS := -std=c++17 -O3 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -DNDEBUG -ffast-math -fforce-addr -fomit-frame-pointer -pthread -pipe -march=native
else
# TODO: use -fsanitize=leak
CXX_FLAGS := -std=c++17  -g3 -O0 -ftrapv -Wall -Wextra -Wno-unused-function -D_LIBCPP_DEBUG -D_GLIBCXX_DEBUG  -pthread -pipe # -pedantic -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused
endif

#  -Wmissing-declarations

MAKE_PID := $(shell echo $$PPID)
JOB_FLAG := $(filter -j%, $(subst -j ,-j,$(shell ps T | grep "^\s*$(MAKE_PID).*$(MAKE)")))
JOBS     := $(subst -j,,$(JOB_FLAG))
ifndef JOBS
JOBS=1
endif

ifdef NO_CUDA
ATEN_FLAGS := -DNO_CUDA=1
CXX_FLAGS := $(CXX_FLAGS) -DNO_CUDA=1
else
ATEN_FLAGS :=
endif

ifdef USE_BOOST
BOOST_LIB := -ldl -lboost_stacktrace_addr2line
BOOST_FLAGS := -DBOOST_ENABLE_ASSERT_DEBUG_HANDLER -DHAVE_BOOST -DBOOST_STACKTRACE_USE_ADDR2LINE
else
BOOST_LIB :=
BOOST_FLAGS :=
endif

.PHONY: test clean test-mnist update-aten install-aten-deps coverage install-cling run-cling

install-aten-deps:
	conda uninstall libgcc; echo ok
	conda install numpy pyyaml mkl setuptools cmake cffi
	conda install -c soumith magma-cuda80
	pip install git+https://github.com/ShigekiKarita/gcovr

install-hdf5-deps:
	conda install hdf5==1.8.17

update-aten:
	git submodule foreach git pull origin master

$(ATEN_ROOT)/lib/libATen.so:
	mkdir -p $(ATEN_ROOT)/../
	cd $(ATEN_ROOT)/../; \
	CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" cmake ../../ATen -DCMAKE_INSTALL_PREFIX=`pwd`/../$(ATEN_ROOT) -DCMAKE_CXX_FLAGS:="-D__STDC_FORMAT_MACROS=1" $(ATEN_FLAGS) ; \
	make install -j$(JOBS)

rapidjson:
	wget -O- https://github.com/Tencent/rapidjson/archive/v$(RAPIDJSON_VER).tar.gz | tar zxvf -
	ln -s `pwd`/rapidjson-1.1.0 rapidjson

data/conf.scp:
	cd data && wget https://github.com/vesis84/kaldi-io-for-python/raw/master/data/conf.scp

data/conf.ark:
	cd data && wget https://github.com/vesis84/kaldi-io-for-python/raw/master/data/conf.ark

%.out: %.cpp $(ATEN_ROOT)/lib/libATen.so
	$(CXX) -o $@ $< $(CXX_FLAGS) $(BOOST_FLAGS) $(INCPATH) $(LIBPATH) $(LIBS) $(BOOST_LIB)

test_serializer.out: test_serializer.cpp $(ATEN_ROOT)/lib/libATen.so
	$(CXX) -o $@ $< $(CXX_FLAGS) $(BOOST_FLAGS) $(INCPATH) $(LIBPATH) $(LIBS) $(BOOST_LIB) $(H5_LIB)

test_mnist.out: test_mnist.cpp
	$(CXX) -o $@ $< $(CXX_FLAGS) $(BOOST_FLAGS) $(INCPATH) $(LIBPATH) $(LIBS) $(BOOST_LIB) $(H5_LIB)

test_ptb.out: test_ptb.cpp
	$(CXX) -o $@ $< $(CXX_FLAGS) $(BOOST_FLAGS) $(INCPATH) $(LIBPATH) $(LIBS) $(BOOST_LIB) $(H5_LIB)

test_kaldi.out: test_kaldi.cpp data/conf.ark data/conf.scp
	$(CXX) -o $@ $<  $(BOOST_FLAGS) $(INCPATH) $(LIBPATH) $(LIBS) $(BOOST_LIB) $(H5_LIB) $(KALDI_FLAGS)

test_wsj.out: test_wsj.cpp rapidjson
	$(CXX) -o $@ $<  -Wall -Wextra -pipe -g3 -O0 -std=c++17 $(BOOST_FLAGS) $(INCPATH) $(LIBPATH) $(LIBS) $(BOOST_LIB) $(H5_LIB) $(KALDI_FLAGS) -Irapidjson/include

test: test_autograd.out test_variable.out test_variable_dsl.out test_chain.out test_dataset.out test_dataloader.out test_sampler.out test_serializer.out test_function.out test_nn_function.out # test_rnn.out
	@ls $^ | LD_LIBRARY_PATH=$(RUNTIME_LIB):$(LD_LIBRARY_PATH) xargs -I{} -n1 -P$(JOBS) sh -c ./{}

data/t10k-images.idx3-ubyte:
	cd data; ./download_mnist.sh

test-mnist: test_mnist.out data/t10k-images.idx3-ubyte
	@LD_LIBRARY_PATH=$(RUNTIME_LIB):$(LD_LIBRARY_PATH) ./test_mnist.out

test-stochastic: test_stochastic.out
	@LD_LIBRARY_PATH=$(RUNTIME_LIB):$(LD_LIBRARY_PATH) ./test_stochastic.out

data/train.txt:
	cd data; ./download_ptb.sh

test-ptb: test_ptb.out data/train.txt
	@LD_LIBRARY_PATH=$(RUNTIME_LIB):$(LD_LIBRARY_PATH) ./test_ptb.out

test-tensor: test_tensor.out
	@LD_LIBRARY_PATH=$(RUNTIME_LIB):$(LD_LIBRARY_PATH) ./test_tensor.out

test-kaldi: test_kaldi.out
	@LD_LIBRARY_PATH=$(RUNTIME_LIB):$(KALDI_ROOT)/src/lib:$(LD_LIBRARY_PATH) ./test_kaldi.out

test-wsj: test_wsj.out
	@LD_LIBRARY_PATH=$(RUNTIME_LIB):$(KALDI_ROOT)/src/lib:$(LD_LIBRARY_PATH) ./test_wsj.out


coverage: clean
coverage: CXX_FLAGS += -coverage -fno-inline -fno-inline-small-functions -fno-default-inline
coverage: test
	gcov -rbf -s ./atnn *.cpp
	lcov -c --no-external -d . -o gcov.info
	# genhtml -o html-gcov gcov.info --ignore-errors source
	mkdir -p html-gcovr
	gcovr -r . -e "test.*.cpp" -e "build/*"  --html --html-details -o html-gcovr/coverage.html

clean:
	rm -fv *.out
	find . -name "*.gcno" | xargs rm -rfv
	find . -name "*.gcda" | xargs rm -rfv
	find . -name "*.gcov" | xargs rm -rfv
	rm -rfv html-gcovr
	rm -fv gcov.info


install-cling:
	# @[ ! -d ./cling ] && echo "install cling from https://root.cern.ch/download/cling/ to ./cling"
	conda install jupyter
	jupyter-kernelspec install --user ./cling/share/cling/Jupyter/kernel/cling-cpp17
	cd ./cling/share/cling/Jupyter/kernel && python setup.py install
	chmod +x ./cling/share/cling/Jupyter/kernel/scripts/jupyter-cling-kernel

run-cling: PATH := $(PWD)/cling/bin:$(PWD)/cling/share/cling/Jupyter/kernel/scripts:$(PATH)
run-cling: CPLUS_INCLUDE_PATH := $(PWD)/$(ATEN_ROOT)/include:$(PWD):$(CONDA_PREFIX):$(CPLUS_INCLUDE_PATH)
run-cling: LD_LIBRARY_PATH := $(PWD)/$(ATEN_ROOT)/lib:$(CONDA_PREFIX)/lib:$(LD_LIBRARY_PATH)
run-cling:
	jupyter notebook --NotebookApp.token='' --ip '*' --open_browser=False


