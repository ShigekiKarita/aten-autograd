#include <ATen/ATen.h>
#include <atnn/serializer.hpp>
#include <atnn/testing.hpp>
#include <stdio.h>
#include <atnn/chain.hpp>

namespace C = atnn::chain;
namespace F = atnn::function;

struct Net : C::Chain {
    C::ChainPtr<C::nn::Linear> linear1, linear2, linear3;

    Net(long n_units=2)
        : linear1(new C::nn::Linear(3, n_units))
        , linear2(new C::nn::Linear(n_units, n_units))
        , linear3(new C::nn::Linear(n_units, 10)) {
        this->chain_dict = {
            ATNN_PARAM(linear1),
            ATNN_PARAM(linear2),
            ATNN_PARAM(linear3)
        };
    }

    auto operator()(atnn::Variable x) {
        auto y1 = C::nn::relu(linear1(x));
        auto y2 = C::nn::relu(linear2(y1));
        return linear3(y2);
    }
};


int main(int argc, char** argv) {
    atnn::test_common(argc, argv, []([[gnu::unused]] auto device) {
            auto path = "./data/tmp.h5";
            remove(path);

            // write via H5Serializer
            auto src = at::CPU(at::kLong).arange(6).view({3, 2});
            atnn::serializer::H5Serializer d1(path);
            d1["test"] = src;

            // read from other serializer
            atnn::serializer::H5Serializer d2(path);
            at::Tensor dst = d2["test"];
            ATNN_ASSERT(atnn::list_eq(src, dst));

            // overwrite "test"
            auto src2 = at::CPU(at::kLong).arange(6).view({3, 2}) * 2;
            d2["test"] = src2;
            at::Tensor dst2 = d2["test"];
            ATNN_ASSERT(atnn::list_eq(src2, dst2));

            // test shape change not supported error
            try {
                auto src3 = at::CPU(at::kLong).arange(9).view({3, 3});
                d2["test"] = src3;
            } catch (atnn::serializer::H5ShapeChangedError&) {
                // ok
            }

            // test model serialization
            path = "./data/tmp_net.h5";
            remove(path);
            atnn::serializer::H5Serializer dnet(path);

            Net net_src;
            auto dict_src = net_src.state_dict();
            dnet.dump(dict_src);
            Net net_dst;
            net_dst.load_state_dict(dnet.load());
            for (auto&& state_dst : net_dst.state_dict()) {
                ATNN_ASSERT(atnn::list_eq(state_dst.second, dict_src[state_dst.first]));
            }
        }, true);
}
