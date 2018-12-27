/**
   \file optimizer.hpp
   \brief gradient based optimizer utility

   \todo support sparse tensors
   \example test_mnist.cpp MNIST example
   \example test_ptb.cpp PTB RNNLM example
*/
#pragma once

#include <unordered_map>
#include "variable.hpp"


namespace atnn {
    /**
        \brief gradient based optimizer utility
        \ingroup atnn::optimizer
    */
    namespace optimizer {

        TList zeros_like(const VList& params) {
            TList ret;
            ret.reserve(params.size());
            for (const auto& p : params) {
                // FIXME
                ret.push_back(p.data().defined() ? p.data().type().zeros_like(p.data()) : at::Tensor{});
            }
            return ret; // NRVO
        }

        struct Optimizer {
            // TODO use VMap instead of VList
            VList parameters;
            std::unordered_map<std::string, double> scalar_dict;
            std::unordered_map<std::string, TList> tensor_dict;

            Optimizer(VList&& parameters, double lr=1.0, double weight_decay=0.0)
                : parameters(std::move(parameters)) {
                this->register_scalars({
                        {"lr", lr},
                        {"weight_decay", weight_decay}
                    });
            }

            virtual ~Optimizer() {}

            void register_scalars(std::unordered_map<std::string, double>&& dict) {
                for (auto&& item : dict) {
                    ATNN_ASSERT(this->scalar_dict.find(item.first) == this->scalar_dict.end());
                    this->scalar_dict[item.first] = item.second;
                }
            }

            void register_tensors(std::vector<std::string>&& name_list) {
                for (auto&& name : name_list) {
                    ATNN_ASSERT(this->tensor_dict.find(name) == this->tensor_dict.end());
                    this->tensor_dict[name] = zeros_like(this->parameters);
                }
            }

            void update() {
                for (size_t i = 0; i < this->parameters.size(); ++i) {
                    auto& p = this->parameters[i];
                    // FIXME
                    if (!p.ptr->grad || !p.ptr->grad->defined()) return;
                    auto weight_decay = this->scalar_dict["weight_decay"];
                    if (weight_decay != 0.0) {
                        p.ptr->grad->data() += weight_decay * p.data();
                    }
                    this->update_one(i);
                }
            }

            double clip_grad_norm(double max_norm, double norm_type=2) {
                double total_norm = 0.0;
                for (const auto& p : this->parameters) {
                    // FIXME
                    if (p.ptr->grad && p.grad().defined()) {
                        total_norm += at::Scalar(p.grad().data().norm(max_norm).pow(max_norm)).toDouble();
                    }
                }
                total_norm = std::pow(total_norm, 1.0 / norm_type);
                auto clip_coef = max_norm / (total_norm + 1e-6);
                if (clip_coef < 1.0) {
                    for (auto& p : this->parameters) {
                        // FIXME
                        if (p.ptr->grad && p.grad().defined()) {
                            p.grad().data() *= clip_coef;
                        }
                    }
                }
                return total_norm;
            }

            virtual void update_one(size_t) = 0;
        };


        struct SGD : Optimizer {
            SGD(VList&& parameters, double lr=1.0, double weight_decay=0.0)
                : Optimizer(std::move(parameters), lr, weight_decay) {
            }

            virtual void update_one(size_t i) override {
                auto& p = this->parameters[i];
                auto lr = this->scalar_dict["lr"];
                p.data() -= lr * p.grad().data();
            }
        };

        struct MomentumSGD : Optimizer {
            MomentumSGD(VList&& parameters, double lr=1.0, double weight_decay=0.0, double momentum=0.0)
                : Optimizer(std::move(parameters), lr, weight_decay) {
                this->register_scalars({{"momentum", momentum}});
                this->register_tensors({"v"});
            }

            virtual void update_one(size_t i) override {
                auto& p = this->parameters[i];
                auto momentum = this->scalar_dict["momentum"];
                auto lr = this->scalar_dict["lr"];
                auto& v = this->tensor_dict["v"][i];
                auto next_v = momentum * v - lr * p.grad().data();
                p.data() += next_v;
                v = next_v;
            }
        };

        struct NesterovSGD : MomentumSGD {
            virtual void update_one(size_t i) override {
                auto& p = this->parameters[i];
                auto momentum = this->scalar_dict["momentum"];
                auto lr = this->scalar_dict["lr"];
                auto& v = this->tensor_dict["v"][i];
                auto next_v = momentum * v - lr * p.grad().data();
                p.data() -= momentum * v + (1.0 - momentum) * next_v;
                v = next_v;
            }
        };

        struct Adam : Optimizer {
            Adam(VList&& parameters, double lr=1e-3, double weight_decay=0.0,
                 double beta1=0.9, double beta2=0.999, double epsilon=1e-8)
                : Optimizer(std::move(parameters), lr, weight_decay) {
                this->register_scalars({{"beta1", beta1}, {"beta2", beta2}, {"epsilon", epsilon}, {"step", 0.0}});
                this->register_tensors({"exp_avg", "exp_avg_sq"});
            }

            virtual void update_one(size_t i) override {
                auto& p = this->parameters[i];
                auto& exp_avg = this->tensor_dict["exp_avg"][i];
                auto& exp_avg_sq = this->tensor_dict["exp_avg_sq"][i];
                auto& beta1 = this->scalar_dict["beta1"];
                auto& beta2 = this->scalar_dict["beta2"];
                auto& step = this->scalar_dict["step"];
                auto grad = p.grad().data();
                ++step;
                exp_avg.mul_(beta1).add_(grad, 1.0 - beta1);
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);
                auto denom = exp_avg_sq.sqrt().add_(this->scalar_dict["epsilon"]);
                auto step_size = this->scalar_dict["lr"] * std::sqrt((1.0 - std::pow(beta2, step)) / (1.0 - std::pow(beta1, step)));
                p.data().addcdiv_(exp_avg, denom, -step_size);
            }
        };

        struct Adadelta : Optimizer {
            Adadelta(VList&& parameters, double lr=1.0, double weight_decay=0.0,
                     double rho=0.9, double epsilon=1e-6)
                : Optimizer(std::move(parameters), lr, weight_decay) {
                this->register_scalars({{"rho", rho}, {"epsilon", epsilon}});
                this->register_tensors({"rms", "acc_delta"});
            }

            virtual void update_one(size_t i) override {
                auto& p = this->parameters[i];
                auto& rms = this->tensor_dict["rms"][i];
                auto rho = this->scalar_dict["rho"];
                auto eps = this->scalar_dict["epsilon"];
                auto& acc_delta = this->tensor_dict["acc_delta"][i];

                auto grad = p.grad().data();
                rms.mul_(rho).addcmul_(grad, grad, 1.0 - rho); // different from pytorch addcmul_ ???
                auto std = rms.add(eps).sqrt_();
                auto delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad);
                p.data() -= this->scalar_dict["lr"] * delta;
                acc_delta.mul_(rho).addcmul_(delta, delta, 1.0 - rho);
            }
        };
    }
}
