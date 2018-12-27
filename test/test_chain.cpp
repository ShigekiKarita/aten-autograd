#include <atnn/function.hpp>
#include <atnn/chain.hpp>
#include <atnn/grad_check.hpp>

namespace F = atnn::function;
namespace C = atnn::chain;

#define PARAM(name) {#name, (name)}


struct Net : C::Chain {
    C::ChainPtr<C::nn::Conv2d> conv1, conv2;
    C::ChainPtr<C::nn::Linear> linear1;

    Net()
        : conv1(new C::nn::Conv2d(4, 2))
        , conv2(new C::nn::Conv2d(2, 2))
        , linear1(new C::nn::Linear(2*3*4, 8)) {
        this->chain_dict = {
            PARAM(conv1),
            PARAM(conv2),
            PARAM(linear1)
        };
    }

    auto operator()(atnn::Variable x) {
        auto y1 = C::nn::sigmoid(conv1(x));
        auto y2 = C::nn::sigmoid(conv2(y1));
        return y2;
    }
};


int main(int argc, char** argv) {
    atnn::test_common(argc, argv, [](auto device) {
            atnn::Variable x(device(at::kFloat).randn({3, 4, 5, 6}));
            Net net;

            // test registered parameters
            ATNN_ASSERT_EQ(net.conv1, net.chain_dict["conv1"]);
            ATNN_ASSERT_EQ(net.linear1, net.chain_dict["linear1"]);
            ATNN_ASSERT_EQ(net.conv1->weight, net.conv1->parameter_dict["weight"]);
            ATNN_ASSERT_EQ(net.conv1->weight.train(), net.conv1->parameter_dict["weight"].train());
            net.conv1->weight.train() = !net.conv1->weight.train();
            ATNN_ASSERT_EQ(net.conv1->weight.train(), net.conv1->parameter_dict["weight"].train());
            for (auto& param : net.conv1->parameter_dict) {
                ATNN_ASSERT_EQ(param.second, net.chain_dict["conv1"]->parameter_dict[param.first]);
            }

            // test set_train(bool)
            bool train = false;
            net.set_train(train);
            ATNN_ASSERT_EQ(net.linear1->weight.train(), train);
            ATNN_ASSERT_EQ(net.linear1->bias.train(), train);
            train = true;
            net.set_train(train);
            ATNN_ASSERT_EQ(net.linear1->weight.train(), train);
            ATNN_ASSERT_EQ(net.linear1->bias.train(), train);

            if (device == at::CUDA) {
                net.toBackend(at::kCUDA);
            }
            auto z = net(x);
            // z.backward(gz);
            auto f0 = [&](auto xs) { return atnn::VList { net(xs[0]) };};
            auto gz = device(at::kFloat).ones(z.sizes());
            atnn::grad_check(f0, {x, net.conv1->weight, net.conv1->bias}, {gz}, 1e-1, 1e-2, 1e-2);

        });
}
