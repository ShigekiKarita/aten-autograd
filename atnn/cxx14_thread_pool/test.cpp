#include "thread_pool.hpp"
#include <iostream>
#include <vector>

using namespace thread_pool;


int main () {
    ThreadPool<> pool(6);
    for (int i = 10; i > 0; --i) {
        pool.enqueue([i] {
                std::this_thread::sleep_for(std::chrono::milliseconds(i*100));
                std::cout << "Simple: " << i << std::endl; });
    }

    ThreadPool<PriorityQueue> ppool(6);
    for (int i = 10; i > 0; --i) {
        ppool.enqueue(i, [i] {
                std::this_thread::sleep_for(std::chrono::milliseconds(i*100));
                std::cout << "Priority:" << i << std::endl; });
    }

}
