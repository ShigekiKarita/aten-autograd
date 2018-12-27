CXX := g++
ATEN_ROOT := build/stage
INCPATH := -I$(ATEN_ROOT)/include -I. -I$(CONDA_PREFIX)/include
LIBPATH := -L$(ATEN_ROOT)/lib -L$(CONDA_PREFIX)/lib
LIBS := -lATen
H5_LIB := -lhdf5 -lhdf5_cpp
RUNTIME_LIB := $(ATEN_ROOT)/lib:$(CONDA_PREFIX)/lib


install-cling:
	# @[ ! -d ./cling ] && echo "install cling from https://root.cern.ch/download/cling/ to ./cling"
	conda install jupyter
	jupyter-kernelspec install --user ./cling/share/cling/Jupyter/kernel/cling-cpp17
	cd ./cling/share/cling/Jupyter/kernel && python setup.py install
	chmod +x ./cling/share/cling/Jupyter/kernel/scripts/jupyter-cling-kernel

run-cling: PATH := $(PWD)/cling/bin:$(PWD)/cling/share/cling/Jupyter/kernel/scripts:$(PATH)
run-cling: CPLUS_INCLUDE_PATH := $(PWD)/..:$(PWD)/$(ATEN_ROOT)/include:$(PWD):$(CONDA_PREFIX):$(CPLUS_INCLUDE_PATH)
run-cling: LD_LIBRARY_PATH := $(PWD)/$(ATEN_ROOT)/lib:$(CONDA_PREFIX)/lib:$(LD_LIBRARY_PATH)
run-cling:
	jupyter notebook --NotebookApp.token='' --ip '*' --open_browser=False


