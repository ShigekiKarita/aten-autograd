#include <atnn/function.hpp>
#include <atnn/testing.hpp>
#include <atnn/grad_check.hpp>
#include <atnn/chain.hpp>


struct Pow : atnn::function::Function<Pow> {
    auto impl_forward(const atnn::TList& x) {
        this->save_for_backward(x);
        return x[0].pow(this->n);
    }

    atnn::VList impl_backward(const atnn::VList& gy) {
        auto&& _x = this->saved_tensors[0];
        return {atnn::Variable(gy[0].data() * _x.pow(this->n - 1) * this->n, false)};
    }
    double n = 2;
    Pow(double n) : n(n) {}
};

struct Add : atnn::function::Function<Add> {
    auto impl_forward(atnn::TList x) {
        return x[0] + x[1];
    }

    atnn::VList impl_backward(atnn::VList gy) {
        return {gy[0], gy[0]};
    }
};


int main(int argc, char** argv) {
    atnn::test_common(argc, argv, [](auto device) {
            {
                at::Tensor d = device(at::kFloat).randn({3, 4});
                at::Tensor gy = device(at::kFloat).randn({3, 4});
                auto d_clone = d.clone();

                auto v0 = atnn::Variable(d * 3);
                auto func = atnn::chain::chain_ptr<Pow>(2);
                auto v1 = func(v0);
                ATNN_ASSERT(atnn::allclose(v1.data(), v0.data().pow(2)));

                // TODO: support retain_grads to multiple backwards
                // v1.backward(gy);
                // ATNN_ASSERT(atnn::allclose(d, d_clone)); // check unchanged
                // ATNN_ASSERT(atnn::allclose(v1.grad(), gy));
                // ATNN_ASSERT(atnn::allclose(v0.grad(), v1.grad() * v0.data() * 2));
                // auto prev_g1 = v1.grad().clone();
                // auto prev_g0 = v0.grad().clone();

                auto add = atnn::chain::chain_ptr<Add>();
                auto v2 = v0 + v1; // add(v0, v1); // v2 = v0 + (v0 * v0) -> dv2/dv0 = 1 + 2 * v0

                v2.clear_grads();
                ATNN_ASSERT(v0.ptr->grad == nullptr);
                ATNN_ASSERT(v1.ptr->grad == nullptr);

                v2.backward(gy);
                ATNN_ASSERT(atnn::allclose(d, d_clone)); // check unchanged
                ATNN_ASSERT(atnn::allclose(v2.grad().data(), gy));
                ATNN_ASSERT(atnn::allclose(v1.grad().data(), gy));
                ATNN_ASSERT(atnn::allclose(v0.grad().data(), gy * (2 * v0.data() + 1), 1e-6));

                // TODO: support retain_grads to multiple backwards
                // auto v2_op = v0 + v1;
                // v2_op.clear_grads();
                // v2_op.backward(gy);
                // ATNN_ASSERT(atnn::allclose(v2.data(), v2_op.data()));
                // ATNN_ASSERT(atnn::allclose(v2.grad(), v2_op.grad()));
                // ATNN_ASSERT(atnn::allclose(v1.grad(), gy));
                // ATNN_ASSERT(atnn::allclose(v0.grad(), gy * (2 * v0.data() + 1), 1e-6));

                auto f = [&](auto x) { return func(x[0]); };
                atnn::grad_check(f, {v0}, {gy});

                auto f1 = [&](auto x) { return add(x[0], func(x[0])); };
                atnn::grad_check(f1, {v0}, {gy});
            }

            {
                at::Tensor d = device(at::kFloat).ones({3, 4});
                auto v0 = atnn::Variable(d * 3);
                auto v1 = atnn::Variable(d * 2);
                auto add = std::make_shared<Add>();
                auto v2 = add->forward(v0, v1);
                ATNN_ASSERT(atnn::allclose(v2.data(), d * 5));
                v2.backward(d);
                ATNN_ASSERT(atnn::allclose(v0.grad().data(), d));
                ATNN_ASSERT(atnn::allclose(v1.grad().data(), d));
            }
        });
}
