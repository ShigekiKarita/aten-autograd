/**
   \file initializer.hpp
   \brief Weight initialization utility
 */
#pragma once

#include <cmath>
#include <ATen/ATen.h>
#include "variable.hpp"
#include "function.hpp"
#include "testing.hpp"

namespace atnn {
    /**
        \brief Weight initialization utility
        \ingroup atnn::initializer
    */
    namespace initializer {
        struct Initializer {
            virtual ~Initializer() {}

            virtual void init(at::Tensor& t) const = 0;

            void operator()(at::Tensor& t) const {
                this->init(t);
            }

            void operator()(Variable& v) const {
                this->init(v.ptr->data);
            }
        };

        struct Uniform : Initializer {
            const double a, b;
            Uniform(double a=0, double b=1) : a(a), b(b) {}

            virtual void init(at::Tensor& t) const override {
                t.uniform_(this->a, this->b);
            }
        };

        struct Normal : Initializer {
            const double mean, std;
            Normal(double mean=0, double std=1) : mean(mean), std(std) {}

            virtual void init(at::Tensor& t) const override {
                t.normal_(this->mean, this->std);
            }
        };

        struct Fan {
            int64_t in, out;

            Fan(const at::Tensor& t) {
                ATNN_ASSERT(t.dim() >= 2);
                if (t.dim() == 2) {
                    this->in = t.size(1);
                    this->out = t.size(0);
                } else {
                    auto in_fmap = t.size(1);
                    auto out_fmap = t.size(0);
                    auto field = t.dim() > 2 ? t[0][0].numel() : 1;
                    this->in = in_fmap * field;
                    this->out = out_fmap * field;
                }
            }
        };

        struct XavierUniform : Initializer {
            double gain;
            XavierUniform(double gain=1) : gain(gain) {}

            virtual void init(at::Tensor& t) const override {
                Fan fan(t);
                auto a = std::sqrt(3.0) * this->gain * std::sqrt(2.0 / (fan.in + fan.out));
                t.uniform_(-a, a);
            }
        };

        struct XavierNormal : Initializer {
            double gain;
            XavierNormal(double gain=1) : gain(gain) {}

            virtual void init(at::Tensor& t) const override {
                Fan fan(t);
                auto std = this->gain * std::sqrt(2.0 / (fan.in + fan.out));
                t.normal_(0, std);
            }
        };

        template <typename T>
        constexpr double gain_of = 1.0;

        template <>
        double gain_of<atnn::function::ReLU> = std::sqrt(2.0);

        template <>
        double gain_of<atnn::function::Tanh> = 5.0 / 3.0;

        template <typename N = atnn::function::ReLU>
        struct KammingUniform : Uniform {
            bool use_fan_in;
            KammingUniform(bool use_fan_in=true) : use_fan_in(use_fan_in) {}

            virtual void init(at::Tensor& t) const override {
                Fan fan(t);
                auto f = this->use_fan_in ? fan.in : fan.out;
                auto a = std::sqrt(3.0) * gain_of<N> / std::sqrt(f);
                t.uniform_(-a, a);
            }
        };

        template <typename N = atnn::function::ReLU>
        struct KammingNormal : Normal {
            bool use_fan_in;
            KammingNormal(bool use_fan_in=true) : use_fan_in(use_fan_in) {}

            virtual void init(at::Tensor& t) const override {
                Fan fan(t);
                auto f = this->use_fan_in ? fan.in : fan.out;
                auto std = gain_of<N> / std::sqrt(f);
                t.normal_(0, std);
            }
        };
    } // namespace initializer
} // namespace atnn
