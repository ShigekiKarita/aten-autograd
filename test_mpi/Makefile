.PHONY: test clean install-mpi install-boost

CXX = mpicxx
MPI_ROOT := $(CONDA_PREFIX)
# MPI_ROOT := /home/skarita/tool/mvapich2_gdr_cuda8.0
MPI_LIB := $(MPI_ROOT)/lib64
# MPI_LIB := $(MPI_ROOT)/lib
MPI_INC := $(MPI_ROOT)/include
PATH := $(MPI_ROOT)/bin:$(PATH)


BOOST_VERSION := 66
BOOST_MINOR := 0
BOOST_ROOT := boost_1_$(BOOST_VERSION)_$(BOOST_MINOR)
BOOST_STAGE := $(BOOST_ROOT)/stage
BOOST_LIB := $(BOOST_STAGE)/lib
BOOST_INC := $(BOOST_STAGE)/include
LD_LIBRARY_PATH := $(BOOST_LIB):$(MPI_LIB):$(MPI_ROOT)/gdrcopy:$(LD_LIBRARY_PATH)


$(BOOST_ROOT).tar.bz2: install-mpi
	wget https://dl.bintray.com/boostorg/release/1.$(BOOST_VERSION).$(BOOST_MINOR)/source/$(BOOST_ROOT).tar.bz2

$(BOOST_ROOT)/.dirstamp: $(BOOST_ROOT).tar.bz2
	tar -xvf $< && touch $@

$(BOOST_STAGE)/lib/libboost_mpi.a: $(BOOST_ROOT)
	cd $(BOOST_ROOT) && ./bootstrap.sh && echo "using mpi ;" >>  project-config.jam && ./b2 --without-python -j10 --prefix=$(PWD)/$(BOOST_STAGE)

install-mpi:
	conda install -y -c mpi4py openmpi

install-boost: $(BOOST_STAGE)/lib/libboost_mpi.a

%.out: %.cpp
	ldd $(BOOST_LIB)/libboost_mpi.so
	$(CXX) $< -o $@ -std=c++11 -I$(BOOST_INC) -L$(BOOST_LIB) -I$(MPI_INC) -L$(MPI_LIB) -lboost_mpi -lboost_serialization -lmpi

test-mpirun: main.out
	which mpirun
	mpirun --prefix $(CONDA_PREFIX) --hostfile hostfile -N 4  ./main.out

test-srun: main.out
	which mpirun
	salloc -p gpuib -N2 -n 8 mpirun ./main.out

test-local: main.out
	which mpirun
	mpirun --hosts $(hostname) ./main.out

clean:
	rm *.out

