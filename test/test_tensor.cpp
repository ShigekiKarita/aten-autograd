#include <iostream>

#include <ATen/ATen.h>
#include <atnn/testing.hpp>


int main(int argc, char** argv) {
    atnn::test_common(argc, argv, [](auto device) {
            auto x = device(at::kFloat).randn({3, 2});
            auto index = device(at::kLong).zeros({2});
            index[0] = 1;
            index[1] = 2;
            auto xs = x.index_select(0, index);

            ATNN_ASSERT(atnn::list_eq(xs[0], x[1]));
            ATNN_ASSERT(atnn::list_eq(xs[1], x[2]));

            std::vector<at::Tensor> vec;
            vec.push_back(x);
            ATNN_ASSERT(vec[0].equal(x));
    });
}
