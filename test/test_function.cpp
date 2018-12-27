#include <atnn/testing.hpp>
#include <atnn/grad_check.hpp>
#include <atnn/chain.hpp>

namespace C = atnn::chain;
namespace F = atnn::function;

int main(int argc, char** argv) {
    atnn::test_common(argc, argv, [](auto device) {
            // test add
            {
                atnn::Variable x = device(at::kFloat).rand({3, 4});
                atnn::Variable y = device(at::kFloat).rand({3, 4});
                auto gy = device(at::kFloat).rand({3, 4});
                atnn::grad_check([](auto xs) { return xs[0] + xs[1]; }, {x, y}, {gy});
            }
            // test sub
            {
                atnn::Variable x = device(at::kFloat).rand({3, 4});
                atnn::Variable y = device(at::kFloat).rand({3, 4});
                auto gy = device(at::kFloat).rand({3, 4});
                atnn::grad_check([](auto xs) { return xs[0] - xs[1]; }, {x, y}, {gy});
            }            
            // test mul
            {
                atnn::Variable x = device(at::kFloat).rand({3, 4});
                atnn::Variable y = device(at::kFloat).rand({3, 4});
                auto gy = device(at::kFloat).rand({3, 4});
                atnn::grad_check([](auto xs) { return xs[0] * xs[1]; }, {x, y}, {gy});

                // test scalar
                atnn::grad_check([](auto xs) { return xs[0] * 2.0; }, {x}, {gy});
                atnn::grad_check([](auto xs) { return 2.0 * xs[0]; }, {x}, {gy});
            }
            // test div
            {
                atnn::Variable x = device(at::kFloat).randn({3, 4});
                atnn::Variable y = device(at::kFloat).randn({3, 4});
                auto gy = device(at::kFloat).randn({3, 4});
                atnn::grad_check([](auto xs) { return xs[0] / xs[1]; }, {x, y}, {gy}, 1e-4, 1e-2);
                // test scalar
                atnn::grad_check([](auto xs) { return xs[0] / 2.0; }, {x}, {gy}, 1e-4, 1e-2);
                atnn::grad_check([](auto xs) { return 2.0 / xs[0]; }, {x}, {gy}, 1e-4, 1e-2);
            }
            // test pow double
            {
                atnn::Variable x = device(at::kFloat).randn({3, 4});
                auto gy = device(at::kFloat).randn({3, 4});
                atnn::grad_check([](auto xs) { return xs[0].pow(2.0); }, {x}, {gy});
            }
            // test pow variable
            {
                atnn::Variable x = device(at::kFloat).rand({3, 4}) + 1.0; // must be positive
                atnn::Variable y = device(at::kFloat).randn({3, 4});
                auto gy = device(at::kFloat).randn({3, 4});
                atnn::grad_check([](auto xs) { return xs[0].pow(xs[1]); }, {x, y}, {gy}, 1e-3, 1e-2);
            }
            // test log variable
            {
                atnn::Variable x = device(at::kFloat).rand({3, 4}) + 1.0; // must be positive
                auto gy = device(at::kFloat).randn({3, 4});
                atnn::grad_check([](auto xs) { return xs[0].log(); }, {x}, {gy}, 1e-3, 1e-2);
            }

            // test sum
            {
                atnn::Variable x = device(at::kFloat).rand({2, 3, 4});
                for (bool keepdim : {true, false}) {
                    for (long dim : {0, 1, 2}) {
                        ATNN_ASSERT(atnn::list_eq(x.sum(dim, keepdim).data(), x.data().sum(dim, keepdim)));
                    }
                }
            }
            // test sum all
            {
                atnn::Variable x = device(at::kFloat).rand({3, 4});
                ATNN_ASSERT(( at::Scalar(x.sum().data()).toDouble() == at::Scalar(x.data().sum()).toDouble() ));
                auto y = x.sum();
                auto gy = device(at::kFloat).randn({1});
                y.backward(gy);
                ATNN_ASSERT(atnn::list_eq(x.grad().data(), gy.expand_as(x.data())));
                atnn::grad_check([](auto xs) { return xs[0].sum(); }, {x}, {gy}, 1e-3, 1e-2);
            }
            // test nested select
            {
                atnn::Variable x = device(at::kFloat).rand({3, 4, 2});
                auto y = x.select(1, 2).select(0, 1);
                ATNN_ASSERT(atnn::list_eq(y.data(), x.data()[1][2]));
                auto gy = device(at::kFloat).rand({2});
                y.backward(gy);
                auto z = device(at::kFloat).zeros({4, 2});
                ATNN_ASSERT(atnn::list_eq(x.grad().data()[0], z));
                ATNN_ASSERT(atnn::list_eq(x.grad().data()[1][2], gy));
                ATNN_ASSERT(atnn::list_eq(x.grad().data()[2], z));
            }
            // test nested select with operator[]
            {
                atnn::Variable x = device(at::kFloat).rand({3, 4, 2});
                auto y = x[1][2];
                ATNN_ASSERT(atnn::list_eq(y.data(), x.data()[1][2]));
                auto gy = device(at::kFloat).rand({2});
                y.backward(gy);
                auto z = device(at::kFloat).zeros({4, 2});
                ATNN_ASSERT(atnn::list_eq(x.grad().data()[0], z));
                ATNN_ASSERT(atnn::list_eq(x.grad().data()[1][2], gy));
                ATNN_ASSERT(atnn::list_eq(x.grad().data()[2], z));
            }
            // test index_select
            {
                atnn::Variable x = device(at::kFloat).rand({3, 2});
                auto index = device(at::kLong).zeros({2});
                index[0] = 1;
                index[1] = 2;
                auto y = x.index_select(0, index);
                ATNN_ASSERT(atnn::list_eq(y.data()[0], x.data()[1]));
                ATNN_ASSERT(atnn::list_eq(y.data()[1], x.data()[2]));
                auto gy = device(at::kFloat).rand(y.sizes());
                y.backward(gy);
                auto z = device(at::kFloat).zeros({2});
                ATNN_ASSERT(atnn::list_eq(x.grad().data()[0], z));
                ATNN_ASSERT(atnn::list_eq(x.grad().data().index_select(0, index), gy));
            }
            // test index_select with operator[]
            {
                atnn::Variable x = device(at::kFloat).rand({3, 2});
                auto index = device(at::kLong).zeros({2});
                index[0] = 1;
                index[1] = 2;
                auto xs = x[index];
                ATNN_ASSERT(atnn::list_eq(xs.data()[0], x.data()[1]));
                ATNN_ASSERT(atnn::list_eq(xs.data()[1], x.data()[2]));
            }
            // test view
            {
                atnn::Variable x = device(at::kFloat).rand({3, 2, 2});
                auto y = x.view({3, 4});
                ATNN_ASSERT(atnn::list_eq(y.data(), x.data().view(y.sizes())));
                auto gy = device(at::kFloat).rand(y.sizes());
                y.backward(gy);
                ATNN_ASSERT(atnn::list_eq(x.grad().data(), gy.view(x.sizes())));
            }
            // test unsqueeze
            {
                atnn::Variable x = device(at::kFloat).rand({3, 2, 2});
                ATNN_ASSERT(atnn::list_eq(x.unsqueeze(0).sizes(), at::IntList{1, 3, 2, 2}));
                ATNN_ASSERT(atnn::list_eq(x.unsqueeze(-4).sizes(), at::IntList{1, 3, 2, 2}));
                ATNN_ASSERT(atnn::list_eq(x.unsqueeze(1).sizes(), at::IntList{3, 1, 2, 2}));
                ATNN_ASSERT(atnn::list_eq(x.unsqueeze(-3).sizes(), at::IntList{3, 1, 2, 2}));
                ATNN_ASSERT(atnn::list_eq(x.unsqueeze(2).sizes(), at::IntList{3, 2, 1, 2}));
                ATNN_ASSERT(atnn::list_eq(x.unsqueeze(-2).sizes(), at::IntList{3, 2, 1, 2}));
                ATNN_ASSERT(atnn::list_eq(x.unsqueeze(3).sizes(), at::IntList{3, 2, 2, 1}));
                ATNN_ASSERT(atnn::list_eq(x.unsqueeze(-1).sizes(), at::IntList{3, 2, 2, 1}));
            }
            // test squeeze
            {
                atnn::Variable x = device(at::kFloat).rand({1, 3, 1, 2, 2, 1, 1});
                ATNN_ASSERT(atnn::list_eq(x.squeeze().sizes(), at::IntList{3, 2, 2}));                

                ATNN_ASSERT(atnn::list_eq(x.squeeze(0).sizes(), at::IntList{3, 1, 2, 2, 1, 1}));
                ATNN_ASSERT(atnn::list_eq(x.squeeze(2).sizes(), at::IntList{1, 3, 2, 2, 1, 1}));
                ATNN_ASSERT(atnn::list_eq(x.squeeze(x.dim()-1).sizes(), at::IntList{1, 3, 1, 2, 2, 1}));
                ATNN_ASSERT(atnn::list_eq(x.squeeze(-2).sizes(), at::IntList{1, 3, 1, 2, 2, 1}));
            }
            // test cat
            {
                atnn::Variable x1 = device(at::kFloat).rand({2, 2});
                atnn::Variable x2 = device(at::kFloat).rand({1, 2});
                atnn::Variable x3 = device(at::kFloat).rand({3, 2});
                std::vector<atnn::Variable> xs = {x1, x2, x3};
                auto y = atnn::chain::cat(xs, 0);
                ATNN_ASSERT(atnn::list_eq(y.data(), at::cat({x1.data(), x2.data(), x3.data()}, 0)));
                auto gy = device(at::kFloat).rand(y.sizes());
                y.backward(gy);
                ATNN_ASSERT(atnn::list_eq(x1.grad().data(), gy.narrow(0, 0, 2)));                
                ATNN_ASSERT(atnn::list_eq(x2.grad().data(), gy.narrow(0, 2, 1)));                
                ATNN_ASSERT(atnn::list_eq(x3.grad().data(), gy.narrow(0, 3, 3)));                
            }
            // test unsqueeze cat
            {
                atnn::Variable x1 = device(at::kFloat).rand({2, 3});
                atnn::Variable x2 = device(at::kFloat).rand({2, 3});
                atnn::Variable x3 = device(at::kFloat).rand({2, 3});
                std::vector<atnn::Variable> xs = {x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1)};
                auto y = atnn::chain::cat(xs, 1);
                // ATNN_ASSERT(atnn::list_eq(y.data(), at::cat({x1.data(), x2.data(), x3.data()}, 0)));
                auto gy = device(at::kFloat).rand(y.sizes());
                y.backward(gy);
                ATNN_ASSERT(atnn::list_eq(x1.grad().data(), gy.select(1, 0)));                
                ATNN_ASSERT(atnn::list_eq(x2.grad().data(), gy.select(1, 1)));                
                ATNN_ASSERT(atnn::list_eq(x3.grad().data(), gy.select(1, 2)));                
            }
            // test split
            {
                atnn::Variable x1 = device(at::kFloat).rand({5, 2});
                auto ys = x1.split(2, 0);
                // at::split is broken but at::native works ?
                ATNN_ASSERT(atnn::list_eq(ys[0].data(), x1.data().narrow(0, 0, 2)));
                ATNN_ASSERT(atnn::list_eq(ys[1].data(), x1.data().narrow(0, 2, 2)));
                ATNN_ASSERT(atnn::list_eq(ys[2].data(), x1.data().narrow(0, 4, 1)));
                auto z = atnn::chain::cat(ys, 0);
                auto gy = device(at::kFloat).rand(z.sizes());
                z.backward(gy);
                ATNN_ASSERT(atnn::list_eq(x1.grad().data(), gy));
            }
            // test chunk
            {
                atnn::Variable x1 = device(at::kFloat).rand({5, 2});
                auto ys = x1.chunk(2, 0);
                // at::split is broken but at::native works ?
                ATNN_ASSERT(atnn::list_eq(ys[0].data(), x1.data().narrow(0, 0, 3)));
                ATNN_ASSERT(atnn::list_eq(ys[1].data(), x1.data().narrow(0, 3, 2)));
                auto z = atnn::chain::cat(ys, 0);
                auto gy = device(at::kFloat).rand(z.sizes());
                z.backward(gy);
                ATNN_ASSERT(atnn::list_eq(x1.grad().data(), gy));            
            }            
            // test transpose
            {
                atnn::Variable x = device(at::kFloat).rand({3, 2, 2});
                auto xt = x.t();
                auto y = x.transpose(0, 2);
                ATNN_ASSERT(atnn::list_eq(xt.data(), x.data().t()));
                ATNN_ASSERT(atnn::list_eq(y.data(), x.data().transpose(0, 2)));

                auto gy = device(at::kFloat).rand(y.sizes());
                y.backward(gy);
                ATNN_ASSERT(atnn::list_eq(x.grad().data(), gy.transpose(2, 0)));
            }
            // test permute
            {
                atnn::Variable x = device(at::kFloat).rand({3, 2, 2});
                auto y = x.permute({2, 0, 1});
                ATNN_ASSERT(atnn::list_eq(y.data(), x.data().permute({2, 0, 1})));
                auto gy = device(at::kFloat).rand(y.sizes());
                y.backward(gy);
                ATNN_ASSERT(atnn::list_eq(x.grad().data(), gy.permute({1, 2, 0})));
            }
            // FIXME : "slice step must be positive"
            // test slice
            {
                // atnn::Variable 
                atnn::Variable x = device(at::kFloat).rand({5, 3, 2});
                auto y = x.slice(0, 1, 4, 1); // [3, 3, 2]
                auto z = y.slice(1, 0, 2, 1); // [3, 2, 2]
                ATNN_ASSERT(atnn::shape_eq(z.sizes(), at::IntList{3, 2, 2}));
                auto gz = device(at::kFloat).rand(z.sizes());
                z.backward(gz);
                ATNN_ASSERT(atnn::list_eq(x.grad().data().slice(0, 1, 4, 1).slice(1, 0, 2, 1), gz));
            }
            // test slice in operator[]
            {
                atnn::Variable x = device(at::kFloat).rand({5, 3, 2});
                auto y = x[{{2, 5, 1}}];
                ATNN_ASSERT(atnn::shape_eq(y.sizes(), at::IntList{3, 3, 2}));
                auto gy = device(at::kFloat).rand(y.sizes());
                y.backward(gy);
                ATNN_ASSERT(atnn::list_eq(x.grad().data().slice(0, 2, 5, 1), gy));
            }
            {
                atnn::Variable x = device(at::kFloat).rand({5, 3, 2});
                auto z = x[{{0, 5, 2}, {1, 3}, {}}];
                ATNN_ASSERT(atnn::shape_eq(z.sizes(), at::IntList{3, 2, 2}));
                auto gz = device(at::kFloat).rand(z.sizes());
                z.backward(gz);
                ATNN_ASSERT(atnn::list_eq(x.grad().data().slice(0, 0, 5, 2).slice(1, 1, 3, 1), gz));
            }
            // test Lambda
            {
                auto add = C::lambda(
                    [](auto&& xs) { return xs[0] + xs[1]; },
                    []([[gnu::unused]] auto&& xs, auto&& gys) { return atnn::VList { gys[0], gys[0] }; }
                );

                atnn::Variable x = device(at::kFloat).rand({3, 4});
                atnn::Variable y = device(at::kFloat).rand({3, 4});
                auto gy = device(at::kFloat).rand({3, 4});
                atnn::grad_check([add](auto xs) mutable { return add(xs[0], xs[1]); }, {x, y}, {gy});
            }
        });
}
