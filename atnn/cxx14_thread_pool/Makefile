.PHONY: test

test.out: test.cpp thread_pool.hpp
	$(CXX) test.cpp -o test.out -std=c++14 -Wall -pthread

test: test.out
	./test.out

