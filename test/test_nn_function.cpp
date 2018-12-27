#include <atnn/testing.hpp>
#include <atnn/grad_check.hpp>
#include <atnn/chain.hpp>

namespace C = atnn::chain;
namespace F = atnn::function;

int main(int argc, char** argv) {
    atnn::test_common(argc, argv, [](auto device) {
        // test sigmoid
        {
            atnn::Variable x = device(at::kFloat).rand({3, 4});
            auto gy = device(at::kFloat).rand({3, 4});
            atnn::grad_check([](auto&& xs) { return C::nn::sigmoid(xs[0]); }, {x}, {gy});
        }
        // test tanh
        {
            atnn::Variable x = device(at::kFloat).rand({3, 4});
            auto gy = device(at::kFloat).rand({3, 4});
            atnn::grad_check([](auto&& xs) { return C::nn::tanh(xs[0]); }, {x}, {gy});
        }
        // test log_softmax
        {
            atnn::Variable x = device(at::kFloat).rand({3, 4});
            auto gy = device(at::kFloat).rand({3, 4});
            atnn::grad_check([](auto&& xs) { return C::nn::log_softmax(xs[0]); }, {x}, {gy}, 1e-3, 1e-3, 1e-3);
        }
        // test softmax
        {
            atnn::Variable x = device(at::kFloat).rand({3, 4});
            auto gy = device(at::kFloat).rand({3, 4});
            atnn::grad_check([](auto&& xs) { return C::nn::softmax(xs[0]); }, {x}, {gy}, 1e-3, 1e-3, 1e-3);
        }
        // test nll_loss
        {
            atnn::Variable x = device(at::kFloat).rand({3, 4}) + 1.0;
            atnn::Variable t = device(at::kLong).ones({3});
            auto gy = x.data().type().ones({1});
            atnn::grad_check([t](auto&& xs) { return C::nn::nll_loss(xs[0], t); }, {x}, {gy}, 1e-3, 1e-3, 1e-3);
        }
        // test relu
        {
            atnn::Variable x = device(at::kFloat).randn({3, 4});
            auto y = C::nn::relu(x);
            auto gy = device(at::kFloat).ones({3, 4});
            y.backward(gy);
            ATNN_ASSERT(atnn::all_bool(y.data() >= 0));
            ATNN_ASSERT(atnn::all_bool((x.grad().data() > 0.0) == (x.data() > 0.0)));
        }
        // test cross_entropy
        {
            atnn::Variable x = device(at::kFloat).rand({3, 4}) + 1.0;
            atnn::Variable t = device(at::kLong).ones({3});
            auto gy = x.data().type().ones({1});
            atnn::grad_check([t](auto&& xs) { return C::nn::cross_entropy(xs[0], t); }, {x}, {gy}, 1e-3, 1e-3, 1e-3);
        }
        // test mse_loss
        {
            atnn::Variable x = device(at::kFloat).rand({3, 4});
            atnn::Variable t = device(at::kFloat).rand({3, 4});
            auto gy = x.data().type().ones({1});
            atnn::grad_check([t](auto&& xs) { return C::nn::mse_loss(xs[0], t); }, {x}, {gy}, 1e-3, 1e-3, 1e-3);
        }
        // test linear
        {
            atnn::Variable x = device(at::kFloat).rand({3, 4});
            atnn::chain::nn::Linear lin(4, 5);
            if (device == at::CUDA) {
                lin.toBackend(at::kCUDA);
            }
            auto gy = x.data().type().ones({3, 5});
            atnn::grad_check([](auto&& xs) { return C::nn::linear(xs[0], xs[1], xs[2]); },
                {x, lin.weight, lin.bias}, {gy}, 1e-3, 1e-3, 1e-3);
        }
    });
}
