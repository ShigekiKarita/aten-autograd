#include <atnn/function.hpp>
#include <atnn/chain.hpp>
#include <atnn/grad_check.hpp>

namespace F = atnn::function;
namespace C = atnn::chain;

#define PARAM(name) {#name, (name)}


template <typename Activation=F::Sigmoid>
struct RNN : C::Chain {
    C::ChainPtr<C::Linear> upward, lateral;

    RNN(long in_features, long out_features)
        : upward(new C::Linear(in_features, out_features))
        , lateral(new C::Linear(out_features, out_features)) {
        this->chain_dict = {
            PARAM(upward),
            PARAM(lateral)
        };
    }

    auto operator()(atnn::Variable xs) {
        atnn::VList hs;
        hs.reserve(xs.size(0));
        for (long t = 0; t < xs.size(0); ++t) {
            auto h = hs.empty()
                ? this->upward(xs[t])
                : this->upward(xs[t]) + this->lateral(hs.back().squeeze(0));
            auto activation = C::chain_ptr<Activation>();
            hs.push_back(activation(h).unsqueeze(0));
        }
        return C::cat(hs, 0);
    }

    auto operator()(const std::vector<atnn::Variable>& xs) {
        atnn::VList hs;
        hs.reserve(xs.size());
        for (long t = 0; t < xs.size(); ++t) {
            auto h = hs.empty()
                ? this->upward(xs[t])
                // FIXME: this lateral connection raises error
                : this->upward(xs[t]) + this->lateral(hs.back().squeeze(0));
            auto activation = C::chain_ptr<Activation>();
            hs.push_back(h.unsqueeze(0));
        }
        return C::cat(hs, 0);
    }

};


int main(int argc, char** argv) {
    atnn::test_common(argc, argv, [=](auto device) {
            long n_time = 3, n_batch = 2, n_input = 2, n_output = 2;
            RNN<F::Tanh> net(n_input, n_output);
            if (device == at::CUDA) {
                net.toBackend(at::kCUDA);
            }

            // std::vector<atnn::Variable> xs;
            // xs.reserve(n_time);
            // for (long t = 0; t < n_time; ++t) {
            //     xs.emplace_back(device(at::kFloat).rand({n_batch, n_input}));
            // }

            atnn::Variable xs = device(at::kFloat).ones({n_time, n_batch, n_input});
            auto ys = net(xs);
            at::IntList expected = {n_time, n_batch, n_output};
            ATNN_ASSERT(atnn::list_eq(ys.sizes(), expected));
            ys.backward(device(at::kFloat).ones_like(ys.data()));
            ATNN_ASSERT(atnn::shape_eq(xs.grad().sizes(), xs.sizes()));
            ys.clear_grads();
            
            auto gys = device(at::kFloat).ones(ys.sizes());
            atnn::grad_check([&](auto args) { return net(args[0]); }, {xs}, {gys}, std::atof(argv[1]));
            // ATNN_ASSERT_SHAPE_EQ(ys[0].sizes(), (at::IntList {3, 4}));

            // atnn::TList gys(3);
            // for (auto& gy: gys) {
            //     gy = device(at::kFloat).ones(ys[0].sizes());
            // }


            // auto& y_last = ys.back();
            // y_last.backward(gys.back());

            // auto f0 = [&](auto xs) { return atnn::VList { net(xs[0]) };};
            // for (auto& w: {net.upward->weight, net.upward->bias, net.lateral->weight, net.lateral->bias}) {
            //     xs.push_back(w);
            // }

            // FIXME: grad_check calls multiple backward! (need to implement retain_grad=true?)
            // auto f = [&](auto xs) { return net({xs[0], xs[1], xs[2]}); };
            // atnn::grad_check(f, xs, gys);
        });
}
