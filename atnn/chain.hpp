/**
    \file chain.hpp
    \brief Implementation of chainable functions and variable's chainable methods
    \example test_chain.cpp
    \example test_rnn.cpp
*/
#pragma once

#include <algorithm>
#include <memory>
#include <vector>
#include <unordered_map>

#include "function.hpp"
#include "nn_function.hpp"
#include "variable.hpp"
#include "testing.hpp"


#define ATNN_PARAM(name) {#name, (name)}

namespace atnn {
    /**
        \brief Autograd chain (computation graph of functions and variables)
        \ingroup atnn::chain
     */
    namespace chain {
        /**
           \brief Chain handles Variable and forwards them into function::Function
           \todo define to_string
        */
        struct Chain {
            std::unordered_map<std::string, std::shared_ptr<Chain>> chain_dict;
            std::unordered_map<std::string, Variable> parameter_dict;
            bool train;

            virtual ~Chain() {}

            virtual void reset_parameters() {
                for (auto& chain : this->chain_dict) {
                    chain.second->reset_parameters();
                }
            }

            virtual void set_train(bool train=true) {
                this->train = train;
                for (auto& param : this->parameter_dict) {
                    auto p = param.second.ptr;
                    if (p) { p->train = train;}
                }

                for (auto& child : this->chain_dict) {
                    child.second->set_train(train);
                }
            }

            /// move parameters data/grad to the specified backend (at::kCPU or at::kCUDA)
            virtual void toBackend(at::Backend b) {
                for (auto m: this->chain_dict) {
                    m.second->toBackend(b);
                }

                for (auto& param: this->parameter_dict) {
                    auto& p = param.second;
                    p.toBackend(b);
                }
            }

            virtual void clear_grads() {
                // local parameters
                for (auto& param : this->parameter_dict) {
                    param.second.ptr->grad.reset();
                }

                // child parameters
                for (auto& child : this->chain_dict) {
                    child.second->clear_grads();
                }
            }

            void parameters(VList& params) {
                // local parameters
                params.reserve(this->parameter_dict.size());
                for (auto& param : this->parameter_dict) {
                    params.push_back(param.second);
                }

                // child parameters
                for (auto& child : this->chain_dict) {
                    child.second->parameters(params);
                }
            }

            /**
               accumulate all children parameters (Variable)

               \todo return iterator instead of vector
             */
            VList parameters() {
                VList params;
                this->parameters(params);
                return params; // NRVO
            }


            using StateDict = std::unordered_map<std::string, at::Tensor>;

            void state_dict(StateDict& sdict, const std::string& prefix="") const {
                for (const auto& param : this->parameter_dict) {
                    sdict[prefix + param.first] = param.second.data();
                }

                for (auto& child : this->chain_dict) {
                    child.second->state_dict(sdict, prefix + child.first + "-");
                }
            }

            StateDict state_dict(const std::string& prefix="/") const {
                StateDict ret;
                this->state_dict(ret, prefix);
                return ret;
            }

            Chain& load_state_dict(const StateDict& sdict, const std::string& prefix="") {
                for (auto& param : this->parameter_dict) {
                    param.second.data().set_(sdict.at(prefix + param.first));
                }

                for (auto& child : this->chain_dict) {
                    child.second->load_state_dict(sdict, prefix + child.first + "-");
                }
                return *this;
            }
        };

        /// \brief trait to determine whether T is Chain
        template <typename T>
        constexpr bool is_chain = std::is_base_of<Chain, T>::value;

        /// \brief Special std::shared_ptr only for Chain enables operator() via pointer
        template <typename T>
        struct ChainPtr : std::shared_ptr<T> {
            static_assert(is_chain<T>, "T should be derived from atnn::Chain");

            using std::shared_ptr<T>::shared_ptr; // inherit ctors
            explicit ChainPtr(std::shared_ptr<T> s) : std::shared_ptr<T>(s) {}

            template <typename ... Args>
            auto operator()(Args&& ... args) {
                return this->get()->forward(std::forward<Args>(args)...);
            }
        };

        /**
           \brief helper class and function to create parameter-less Chain of function::Function
         */
        template <typename F>
        struct Func : Chain {
            // static_assert(function::is_function<F>, "F should be derived from atnn::Function<F>");
            F f;
            using R = typename std::decay<decltype(*f())>::type;
            Func(F&& f) : f(std::forward<F>(f)) {}

            /// for initializer list {v1, v2, v3}
            auto forward(std::vector<Variable>&& args) {
                return std::shared_ptr<R>(f())->forward(args);
            }

            template <typename ... Args>
            auto forward(Args&& ... args) {
                return std::shared_ptr<R>(f())
                    ->forward(std::forward<Args>(args)...);
            }
        };

        /// \brief functional chain wrapper of Func
        template <typename F>
        auto chain_ptr() {
            auto&& f = []() { return new F(); };
            using L = typename std::decay<decltype(f)>::type;
            using FF = Func<L>;
            return static_cast<ChainPtr<FF>>(std::make_shared<FF>(std::forward<L>(f)));
        }

        /// \brief functional chain wrapper of Func
        template <typename F, typename ... Args>
        auto chain_ptr(Args&& ... args) {
            auto&& f = [args...]() mutable { return new F(std::forward<Args>(args)...); };
            using L = typename std::decay<decltype(f)>::type;
            using FF = Func<L>;
            return static_cast<ChainPtr<FF>>(std::make_shared<FF>(std::forward<L>(f)));
        }

        template <typename Forward, typename Backward>
        auto lambda(Forward&& fwd, Backward&& bwd, bool save_tensors=true) {
            using L = function::Lambda<typename std::decay<Forward>::type, typename std::decay<Backward>::type>;
            return chain_ptr<L>(std::forward<Forward>(fwd), std::forward<Backward>(bwd), save_tensors);
        }        


        /*
          Function-based Chain implementations without parameters
          \todo implement more
         */

        /// \brief functional chain wrapper of function::Cat
        /// \todo enable double backward (implement narrow or slice)
        template <typename VS=std::vector<Variable>>
        auto cat(VS&& xs, int64_t dim) {
            return chain_ptr<function::shape::Cat>(dim)(xs);
        }

    } // namespace chain

    inline Variable fit_shape(Variable x, at::IntList s) {
        if (shape_eq(x.sizes(), s)) return x;
        auto ret = x.data();
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == 1) {
                ret = ret.slice(i, 0, 1);
            }
        }
        return {ret, x.train()};
    }

    /**
       Operator overloading for Variables
     */
    Variable operator+(const Variable& lhs, const Variable& rhs) {
        auto ls = lhs.sizes();
        auto rs = rhs.sizes();
        return chain::lambda(
            [](auto&& xs) { return xs[0] + xs[1]; },
            [ls, rs](auto&& xs, auto&& gys) {
                return VList { fit_shape(gys[0], ls), fit_shape(gys[0], rs) }; }
        )(lhs, rhs);
    }

    auto operator-(const Variable& v) {
        return chain::lambda(
            [](auto&& xs) { return -xs[0]; },
            [](auto&& xs, auto&& gys) { return -gys[0]; }
        )(v);
    }

    auto operator-(const Variable& lhs, const Variable& rhs) {
        return chain::lambda(
            [](auto&& xs) { return xs[0] - xs[1]; },
            [](auto&& xs, auto&& gys) { return VList { gys[0],  -gys[0]}; }
        )(lhs, rhs);
    }

    Variable operator-(at::Scalar lhs, const Variable& rhs) {
        return chain::lambda(
            [lhs](auto&& xs) { return lhs - xs[0]; },
            [lhs](auto&& xs, auto&& gys) { return -gys[0]; }
        )(rhs);
    }

    auto operator*(const Variable& lhs, const Variable& rhs) {
        return chain::lambda(
            [](auto&& xs) { return xs[0] * xs[1]; },
            [](auto&& xs, auto&& gys) { return VList { gys[0] * xs[1],  gys[0] * xs[0] }; },
            true // save_tensors
        )(lhs, rhs);
    }

    Variable operator*(const Variable& lhs, at::Scalar rhs) {
        return chain::lambda(
            [rhs](auto&& xs) { return xs[0] * rhs; },
            [rhs](auto&& xs, auto&& gys) { return gys[0] * rhs; }
        )(lhs);
    }

    Variable operator*(at::Scalar lhs, const Variable& rhs) {
        return rhs * lhs;
    }

    auto operator/(const Variable& lhs, const Variable& rhs) {
        return chain::lambda(
            [](auto&& xs) { return xs[0] / xs[1]; },
            [](auto&& xs, auto&& gys) {
                /// \todo fuse gxs[1] when gys[0].train() == false
                return VList { gys[0] / xs[1],  -gys[0] * xs[0] / (xs[1] * xs[1]) }; },
            true // save_tensors
        )(lhs, rhs);
    }

    Variable operator/(const Variable& lhs, double rhs) {
        return lhs * (1.0 / rhs);
    }

    Variable operator/(at::Scalar lhs, const Variable& rhs) {
        return lhs * rhs.pow(-1.0);
    }

    Variable Variable::pow(double e) const {
        return chain::lambda(
            [e](auto&& xs) { return xs[0].pow(e); },
            [e](auto&& xs, auto&& gys) {
                return VList {gys[0] * xs[0].pow(e - 1.0) * e}; },
            true // save_tensors
        )(*this);
    }

    Variable Variable::pow(Variable e) const {
        return chain::lambda(
            [e](auto&& xs) { return xs[0].pow(xs[1]); },
            [e](auto&& xs, auto&& gys) {
                /// \todo save y = x0.pow(x1) and reuse
                return VList {gys[0] * xs[0].pow(xs[1] - 1) * xs[1], 
                              gys[0] * xs[0].pow(xs[1]) * xs[0].log() }; },
            true // save_tensors
        )(*this, e);
    }

    Variable Variable::log() const {
        return chain::lambda(
            [](auto&& xs) { return xs[0].log(); },
            [](auto&& xs, auto&& gys) { return gys[0] / xs[0]; },
            true // save_tensors
        )(*this);
    }

    Variable Variable::sum(int64_t dim, bool keepdim) const {
        /// \todo enable double backward (after expand/view/permute)
        return chain::chain_ptr<function::math::Sum>(dim, keepdim)(*this);
    }

    Variable Variable::sum() const {
        /// \todo enable double backward (after expand)
        return chain::chain_ptr<function::math::SumAll>()(*this);
    }

    Variable Variable::select(int64_t dim, int64_t index) const {
        /// \todo enable double backward (implement assign)
        /// \todo only save sizes not tensor
        return chain::lambda(
            [dim, index](auto&& xs) { return xs[0].select(dim, index); },
            [dim, index](auto&& xs, auto&& gys) {
                auto data = gys[0].data().type().zeros(xs[0].sizes());
                data.transpose_(0, dim);
                data[index] = gys[0].data();
                data.transpose_(0, dim);
                return Variable(data, false); },
            true
        )(*this);
    }

    Variable Variable::index_select(int64_t dim, at::Tensor index) const {
        /// \todo enable double backward (implement assign)
        /// \todo only save sizes not tensor
        return chain::lambda(
            [dim, index](auto&& xs) { return xs[0].index_select(dim, index); },
            [dim, index](auto&& xs, auto&& gys) {
                auto data = gys[0].data().type().zeros(xs[0].sizes());
                data.index_copy_(dim, index, gys[0].data());
                return Variable(data, false); },
            true
        )(*this);
    }

    Variable Variable::view(at::IntList sizes, const char* debug) const {
        /// \todo only save sizes not tensor
        return chain::lambda(
            [sizes](auto&& xs) { return xs[0].contiguous().view(sizes); },
            [debug](auto&& xs, auto&& gys) { 
                // std::cout << "view-backward:" << debug << gys[0].sizes() << "->" << xs[0].sizes() << std::endl;
                return gys[0].contiguous().view(xs[0].sizes()); },
            true
        )(*this);
    }

    VList Variable::split(int64_t split_size, int64_t dim) {
        return chain::lambda(
            [split_size, dim](auto&& xs) {
                ATNN_ASSERT_EQ(xs.size(), 1);
                auto self = xs[0];
                int64_t dim_size = self.size(dim);
                int64_t num_splits = (dim_size + split_size - 1) / split_size;
                std::vector<at::Tensor> splits(num_splits);
                int64_t last_split_size = split_size - (split_size * num_splits - dim_size);
                for (int64_t i = 0; i < num_splits; ++i) {
                    auto length = i < num_splits - 1 ? split_size : last_split_size;
                    splits[i] = self.narrow(dim, i * split_size, length);
                }
                return splits;
            },
            [dim](auto&&, auto&& gys) { return chain::cat(gys, dim); }
        )(*this);
    }

    Variable Variable::transpose(int64_t dim1, int64_t dim2) const {
        return chain::lambda(
            [dim1, dim2](auto&& xs) { return xs[0].transpose(dim1, dim2); },
            [dim1, dim2](auto&& xs, auto&& gys) { return gys[0].transpose(dim2, dim1); }
        )(*this);
    }

    Variable Variable::permute(at::IntList dims) const {
        return chain::lambda(
            [dims](auto&& xs) { return xs[0].permute(dims); },
            [dims](auto&&, auto&& gys) {
                ATNN_ASSERT_EQ(gys.size(), 1);
                std::vector<int64_t> rev(dims.size());
                for (int64_t i = 0; i < dims.size(); ++i) {
                    rev[dims[i]] = i;
                }
                return gys[0].permute(rev);
            }
        )(*this);
    }

    Variable Variable::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
        /// \todo enable double backward (implement assign)
        /// \todo only save sizes not tensor
        return chain::lambda(
            [start, end, step, dim](auto&& xs) { return xs[0].slice(dim, start, end, step); },
            [start, end, step, dim](auto&& xs, auto&& gys) {
                auto data = gys[0].data().type().zeros_like(xs[0]);
                data.slice(dim, start, end, step) = gys[0].data();
                return Variable(data, false);
            },
            true // save tensors
        )(*this);
        // return chain::chain_ptr<function::shape::Slice>(start, end, step, dim)(*this);
    }

    namespace chain {
        namespace nn {
            inline static auto sigmoid(const Variable& input) {
                return chain_ptr<function::nn::Sigmoid>()(input);
            }

            inline static auto tanh(const Variable& input) {
                return chain_ptr<function::nn::Tanh>()(input);
            }

            inline static auto log_softmax(const Variable& input, int64_t dim = 1) {
                return chain_ptr<function::nn::LogSoftmax>(dim)(input);
            }

            inline static auto softmax(const Variable& input, int64_t dim = 1) {
                return chain_ptr<function::nn::LogSoftmax>(dim)(input);
            }

            inline static auto threshold(const Variable& input, double threshold, double value) {
                return chain_ptr<function::nn::Threshold>(threshold, value)(input);
            }

            inline static auto relu(const Variable& input) {
                return threshold(input, 0.0, 0.0);
            }

            inline static auto nll_loss(Variable logp, Variable target,
                               bool size_average=true, int64_t ignore_index=-1, bool reduce=true,
                               at::Tensor weight={}, at::Tensor total_weight={}) {
                return chain_ptr<function::nn::NLLLoss>(size_average, ignore_index, reduce, weight, total_weight)(logp, target);
            }

            inline static auto mse_loss(Variable x, Variable t, bool size_average=true, bool reduce=true) {
                return chain_ptr<function::nn::MSELoss>(size_average, reduce)(x, t);
            }

            /// \brief This criterion combines `LogSoftmax` and `NLLLoss` in one single function
            inline static auto cross_entropy(Variable input, Variable target,
                               bool size_average=true, int64_t ignore_index=-1, bool reduce=true,
                               at::Tensor weight={}, at::Tensor total_weight={}) {
                auto logp = log_softmax(input);
                auto nll = nll_loss(logp, target, size_average, ignore_index, reduce, weight, total_weight);
                return nll;
            }

            /*
              Function-based Chain implementations with parameters
              computes x.mm(w.t())
             */
            static inline auto linear(Variable x, Variable w, Variable b={}) {
                return lambda(
                    [](auto&& xs) {
                        ATNN_ASSERT_EQ(xs.size(), 3);
                        auto x = xs[0];
                        auto w = xs[1];
                        auto b = xs[2];
                        if (b.defined()) {
                            return b.addmm(x, w.t());
                        } else {
                            return x.mm(w.t());
                        }
                    },
                    [](auto&& xs, auto&& gys) {
                        ATNN_ASSERT_EQ(gys.size(), 1);
                        auto gy = gys[0];
                        auto x = xs[0];
                        auto w = xs[1];
                        auto b = xs[2];
                        VList gxs;
                        gxs.push_back(linear(gy, w.t())); // gx
                        gxs.push_back(linear(gy.t(), x.t())); // gw
                        gxs.push_back(b.defined() ? gy.sum(0) : Variable({}, gy.train()));
                        return gxs;
                    },
                    true 
                )(x, w, b);
            }

            static inline auto mm(Variable a, Variable b) {
                return linear(a, b.transpose(0, 1));
            }


            /// \brief Linear (Affine) transformation chain
            struct Linear : Chain {
                atnn::Variable weight, bias;

                Linear(int64_t in_features, int64_t out_features, bool use_bias=true)
                    : weight(CPU(at::kFloat).zeros({out_features, in_features}))
                    , bias(use_bias ? CPU(at::kFloat).zeros({out_features}) : at::Tensor{}) {
                    this->reset_parameters();
                    this->parameter_dict = {ATNN_PARAM(weight), ATNN_PARAM(bias)};
                }

                virtual void reset_parameters() override {
                    auto std = 1.0 / std::sqrt(static_cast<double>(weight.data().size(1)));
                    this->weight.data().uniform_(-std, std);
                    if (this->bias.defined()) { this->bias.data().uniform_(-std, std); }
                }

                auto forward(const Variable& x) {
                    return linear(x, this->weight, this->bias);
                }
            };

            /** 
             * \todo implement conv2d backward with deconv2d like
             * \ref https://github.com/chainer/chainer/blob/v3.1.0/chainer/functions/connection/convolution_2d.py
             * \ref 
             */
            /// \brief Convolution on 2D feature-map chain
            struct Conv2d : Chain {
                int64_t in_channels, out_channels;
                at::IntList kernel_size, stride, padding;
                atnn::Variable weight, bias;

                Conv2d(int64_t in_channels, int64_t out_channels, at::IntList kernel_size={3, 3},
                       at::IntList stride={1, 1}, at::IntList padding={0, 0}, bool use_bias=true)
                    : in_channels(in_channels)
                    , out_channels(out_channels)
                    , kernel_size(kernel_size)
                    , stride(stride)
                    , padding(padding)
                    , weight(CPU(at::kFloat).zeros({out_channels, in_channels,
                                    kernel_size[0], kernel_size[1]}) * 0.01)
                    , bias(use_bias ? CPU(at::kFloat).zeros(out_channels) : at::Tensor{}) {
                    this->reset_parameters();
                    this->parameter_dict = {ATNN_PARAM(weight), ATNN_PARAM(bias)};
                }

                virtual void reset_parameters() override {
                    auto n = this->in_channels;
                    for (auto k : this->kernel_size) {
                        n *= k;
                    }
                    auto std = 1.0 / std::sqrt(static_cast<double>(n));
                    this->weight.data().uniform_(-std, std);
                    if (this->bias.defined()) { this->bias.data().uniform_(-std, std); }
                }

                auto forward(const Variable& x) {
                    return std::make_shared<function::nn::Conv2d>(out_channels, kernel_size, stride, padding)
                        ->forward(x, weight, bias);
                }
            };

            /**
                \brief ID embedding layer (cannot backprop through IDs)
                \todo support padding idx
            */
            struct Embedding : Chain {
                int64_t num_embeddings, embedding_dim;
                Variable weight;
                Embedding(int64_t num_embeddings, int64_t embedding_dim)
                    : num_embeddings(num_embeddings), embedding_dim(embedding_dim)
                    , weight(CPU(at::kFloat).zeros({num_embeddings, embedding_dim})) {
                    this->parameter_dict = {{"weight", weight}};
                    this->reset_parameters();
                }

                virtual void reset_parameters() override {
                    this->weight.data().normal_(0, 1);
                }

                auto forward(const Variable& x) {
                    ATNN_ASSERT(x.data().dim() == 1);
                    return weight[x.data()];
                }
            };

            /// \brief gate labels for LSTM
            static const std::array<std::string, 4> lstm_gates = {{"i", "f", "o", "c"}};

            /**
                \brief Int64_T short term memory (LSTM) style RNN chain
                \todo implement fused/cudnn version.

                CudnnRNN's flatten_parameters in Pytorch (1)
                \ref https://github.com/pytorch/pytorch/blob/a9ec4ee7423a477a3901ceb41b98462b4d164810/torch/nn/modules/rnn.py#L114
                and copyParams in Pytorch(2)
                \ref https://github.com/pytorch/pytorch/blob/ceb4f84d12304d03a6a46693e54390869c0c208e/torch/backends/cudnn/rnn.py#L180

                RNNFused kernel implementation in Torch
                \ref https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/FusedRNNKernel.cu

                RNNSplitParams in Theano
                \ref https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/FusedRNNKernel.cu
            */
            struct LSTM : Chain {
                int64_t in_features, out_features;
                std::unordered_map<std::string, ChainPtr<Linear>> gates;

                LSTM(int64_t in_features, int64_t out_features)
                    : in_features(in_features), out_features(out_features) {
                    for (auto&& xh : {"x", "h"}) {
                        for (const auto& gates : lstm_gates) {
                            ChainPtr<Linear> ptr(new Linear(xh == "x" ? in_features : out_features, out_features));
                            this->gates[xh + gates] = ptr;
                            this->chain_dict[xh + gates] = ptr;
                        }
                    }
                    this->reset_parameters();
                }

                /// \ref https://github.com/pytorch/pytorch/blob/a9ec4ee7423a477a3901ceb41b98462b4d164810/torch/nn/modules/rnn.py#L126
                virtual void reset_parameters() override {
                    auto std = 1.0 / std::sqrt(this->out_features);
                    for (auto& c : this->chain_dict) {
                        for (auto& p : c.second->parameter_dict) {
                            p.second.data().uniform_(-std, std);
                        }
                    }
                }

                auto forward(const Variable& x, const Variable& h={}, const Variable& c={}) {
                    std::unordered_map<std::string, Variable> g_outs;
                    for (const auto& g : lstm_gates) {
                        auto x_ = this->gates["x" + g](x);
                        g_outs[g] = h.defined() ? x_ + this->gates["h" + g](h) : x_;
                    }
                    auto c_ = sigmoid(g_outs["i"]) * tanh(g_outs["c"]);
                    auto c_next = c.defined() ? sigmoid(g_outs["f"]) * c + c_ : c_;
                    auto h_next = sigmoid(g_outs["o"]) * tanh(c_next);
                    return std::make_tuple(h_next, c_next); 
                } 
            };
        } // namespace nn
    } // namespace chain
    
} // namespace atnn
