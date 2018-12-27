/**
   \file function.hpp
   \brief Autograd function as a computation graph edge

   \example test_function.cpp library Function<Impl> gradient check
   \example test_autograd.cpp user-defined Function<Impl> example
*/
#pragma once

#include <type_traits>
#include <algorithm>
#include <functional>
#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <iostream>

#include "variable.hpp"
#include "testing.hpp"
#include "device.hpp"


namespace atnn {

    /**
     * \brief autograd (forward/backward) function namespace
     *
     * \ingroup function
     */
    namespace function {
        /**
           Function defines forward/backward ops to input/output variables
           without variable members
        */
        template <class Derived>
        struct Function : FunctionBase {
            Derived* dthis = static_cast<Derived*>(this);
            Variable::Map<Variable> grad_outputs;
            bool train = true;

            virtual void register_grad_output(Variable v, const Variable& grad) override {
                if (this->grad_outputs.count(v) == 0 || !this->grad_outputs.at(v).defined()) {
                    this->grad_outputs[v] = grad;
                } else {
                    this->grad_outputs[v] = grad + this->grad_outputs[v];
                }
            }

            bool saficient_grad_outputs() const {
                return std::all_of(this->vrets.begin(), this->vrets.end(),
                                   [this](const auto& v) { return this->grad_outputs.count(v) > 0; });
            }

            /// \brief called at Variable::backward in variable.hpp
            virtual VList compute_grad_inputs() override {
                ATNN_ASSERT(this->vrets.size() > 0);
                if (this->saficient_grad_outputs()) {
                    // reorder
                    VList grads;
                    grads.reserve(this->vrets.size());
                    for (auto&& v : this->vrets) {
                        grads.push_back(std::move(this->grad_outputs[v]));
                    }

                    /// \todo add retain grads option
                    this->grad_outputs.clear();
                    this->vrets.clear();
                    return this->backward(std::move(grads));
                } else if (grad_outputs.size() > this->vrets.size()) {
                    throw_with_trace(std::runtime_error("too many grad outputs"));
                }
                return {};
            }

            virtual ~Function() {}

            auto set_one_varg(Variable v) {
                this->vargs.push_back(v);
                return v.data();
            }

            template <typename ... Args>
            using first_type_of =  typename std::tuple_element<0, std::tuple<Args...>>::type;

            template <typename ... Args>
            using enable_unless_vlist = std::enable_if_t<!std::is_same<VList&, first_type_of<Args...>>::value>;

            template <class ... Args, typename T = enable_unless_vlist<Args...>>
            TList set_vargs(Args&& ... args) {
                return { this->set_one_varg(std::forward<Args>(args))... };
            }

            auto set_vargs(const VList& vs) {
                std::vector<at::Tensor> ret;
                ret.reserve(vs.size());
                for (const auto& v : vs) {
                    this->vargs.push_back(v);
                    ret.push_back(v.data());
                }
                return ret;
            }

            auto set_vargs(VList&& vs) {
                this->vargs = std::move(vs);
                std::vector<at::Tensor> ret;
                ret.reserve(vs.size());
                for (const auto& v : this->vargs) {
                    ret.push_back(v.data());
                }
                return ret;
            }

            auto set_vrets(const at::Tensor& t) {
                auto v = Variable(t, this->train);
                if (this->train) {
                    v.set_function(shared_from_this());
                    this->vrets.push_back(v);
                    ATNN_ASSERT(this->vrets.size() > 0);
                }
                return v;
            }

            auto set_vrets(const TList& ts) {
                ATNN_ASSERT(ts.size() > 0);
                VList vs;
                vs.reserve(ts.size());
                for (const auto& t: ts) {
                    vs.emplace_back(t, this->train);
                }
                if (this->train) {
                    this->vrets = std::move(vs);
                    for (auto& v: this->vrets) {
                        v.set_function(this->shared_from_this());
                    }
                    ATNN_ASSERT(this->vrets.size() > 0);
                    return this->vrets;
                }
                return vs;
            }

            template <class ... Args>
            auto forward(Args&& ... args) {
                // static_assert(!std::is_same<VList&, first_type_of<Args...>>::value, "not vector");
                this->vargs.clear();
                this->vrets.clear();
                this->grad_outputs.clear();
                auto&& trets = dthis->impl_forward(this->set_vargs(std::forward<Args>(args)...));
                this->train = this->vargs.front().train();
                // FXIME:
                // for (auto& v: this->vargs) {
                //     if (v.train() != this->train) {
                //         throw_with_trace(std::runtime_error("Variable.train=true/false cannot be mixed"));
                //     }
                // }
                return set_vrets(std::forward<decltype(trets)>(trets)); // RVO
            }

            VList to_list(Variable&& v) {
                return {v};
            }

            VList to_list(VList&& v) {
                return std::move(v);
            }

            VList backward(const VList& grads) {
                // bool train = all_train(grads);
                // this->saved_variables.reserve(this>saved_tensors.size());
                // for (auto&& t : this->saved_tensors) {
                //     this->saved_variables.emplace_back(t, train);
                // }
                return to_list(std::move(dthis->impl_backward(grads)));
            }

            void save_for_backward(const TList& tensors, const std::vector<at::IntList>& sizes={}){
                // FIXME: rethink this
                bool train = true;
                for (auto&& v: this->vargs) {
                    train &= v.train();
                    if (!train) return;
                }
                this->saved_tensors = tensors;
                this->saved_sizes = sizes;
            }

            void toBackend(at::Backend b) override {
                for (auto& t: this->saved_tensors) {
                    t = t.toBackend(b);
                }
            }
        };

        template <typename T>
        constexpr bool is_function = std::is_base_of<FunctionBase, T>::value;

        template <typename Forward, typename Backward>
        struct Lambda : Function<Lambda<Forward, Backward>> {
            auto impl_forward(const TList& xs) {
                if (this->save_tensors) this->save_for_backward(xs);
                return this->fwd(xs);
            }

            auto impl_backward(const VList& gys) {
                return this->bwd(this->saved_tensors, gys);
            }

            bool save_tensors;
            Forward fwd;
            Backward bwd;

            Lambda(Forward&& fwd, Backward&& bwd, bool save_tensors)
                : fwd(std::forward<Forward>(fwd))
                , bwd(std::forward<Backward>(bwd))
                , save_tensors(save_tensors) {}
        };

        template <typename Forward, typename Backward>
        Lambda<typename std::decay<Forward>::type,
               typename std::decay<Backward>::type>
        make_lambda(Forward&& fwd, Backward&& bwd, bool save_tensors=false) {
            return { std::forward<Forward>(fwd), std::forward<Backward>(bwd), save_tensors };
        }


        namespace math {

            struct Sum : Function<Sum> {
                int64_t dim;
                bool keepdim;

                auto impl_forward(TList xs) {
                    ATNN_ASSERT_EQ(xs.size(), 1);
                    this->save_for_backward({}, {xs[0].sizes()});
                    return xs[0].sum(dim, keepdim);
                }

                VList impl_backward(VList gys) {
                    ATNN_ASSERT_EQ(gys.size(), 1);
                    auto&& x_size = this->saved_sizes[0];
                    std::vector<int64_t> expand_size = {at::Scalar(x_size[dim]).toLong()};
                    expand_size.reserve(x_size.size());
                    std::copy(x_size.begin(), x_size.end(), std::back_inserter(expand_size));
                    auto gy = gys[0].data().expand(expand_size);
                    // gy  [0:e, 1:0, 2:1, ..., e:e-1, e+1:e+1, ...]
                    // gy_ [1:0, 2:1, ..., e:e-1, 0:e, e+1:e+1, ...]
                    std::vector<int64_t> dims;
                    dims.reserve(gy.dim());
                    for (int64_t i = 0; i < gy.dim(); ++i) {
                        int64_t n = 0;
                        if (i < this->dim) { n = i + 1; }
                        else if (i == this->dim) { n = 0; }
                        else { n = i; }
                        dims.push_back(n);
                    }
                    gy = gy.permute(dims);
                    if (keepdim) {
                        return {gy.view(x_size)};
                    }
                    return { Variable(gy, false) }; // gys[0].expand
                }

                Sum(int64_t dim, bool keepdim=false) : dim(dim), keepdim(keepdim) {}
            };

            struct SumAll : Function<SumAll> {
                auto impl_forward(TList xs) {
                    ATNN_ASSERT_EQ(xs.size(), 1);
                    this->save_for_backward({}, {xs[0].sizes()});
                    return xs[0].sum();
                }

                VList impl_backward(VList gys) {
                    ATNN_ASSERT_EQ(gys.size(), 1);
                    auto&& x_size = this->saved_sizes[0];
                    return { Variable(gys[0].data().expand(x_size), false) };
                }
            };

        } // namespace math

        namespace shape {
            struct Cat : Function<Cat> {
                auto impl_forward(const TList& xs) {
                    this->save_for_backward(xs);
                    return at::cat(xs, this->dim);
                }

                VList impl_backward(const VList& gys) {
                    ATNN_ASSERT_EQ(gys.size(), 1);
                    auto xs = this->saved_tensors;
                    VList gxs;
                    gxs.reserve(xs.size());
                    int64_t index = 0;
                    for (auto&& x : xs) {
                        int64_t size = x.size(this->dim);
                        gxs.emplace_back(gys[0].data().narrow(this->dim, index, size), gys[0].train());
                        index += size;
                    }
                    return gxs;
                }

                int64_t dim;
                Cat(int64_t dim) : dim(dim) {}
            };

        } // namespace shaping
    } // namespace function
} // namespace atnn
