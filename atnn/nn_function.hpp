#include "function.hpp"

namespace atnn {
    namespace function {
        /**
         * \brief wrapper of torch forward/backward implementation
         * \ingroup nn
         */
        namespace nn {
            struct Sigmoid : Function<Sigmoid> {
                auto impl_forward(const TList& xs) {
                    ATNN_ASSERT_EQ(xs.size(), 1);
                    auto y = at::_sigmoid_forward(xs[0]);
                    this->save_for_backward({y});
                    return y;
                }

                auto impl_backward(const VList& gy) {
                    ATNN_ASSERT_EQ(gy.size(), 1);
                    ATNN_ASSERT_MSG(!gy[0].train(), "this function is not double backwardable yet");
                    auto y = this->saved_tensors[0];
                    ATNN_ASSERT_SHAPE_EQ(gy[0].sizes(), y.sizes());
                    return Variable(at::_sigmoid_backward(gy[0].data(), y), false);
                }
            };

            struct Tanh : Function<Tanh> {
                auto impl_forward(const TList& xs) {
                    ATNN_ASSERT_EQ(xs.size(), 1);
                    auto y = at::_tanh_forward(xs[0]);
                    this->save_for_backward({y});
                    return y;
                }

                auto impl_backward(const VList& gy) {
                    ATNN_ASSERT_EQ(gy.size(), 1);
                    ATNN_ASSERT_MSG(!gy[0].train(), "this function is not double backwardable yet");
                    auto y = this->saved_tensors[0];
                    ATNN_ASSERT_SHAPE_EQ(gy[0].sizes(), y.sizes());
                    return Variable(at::_tanh_backward(gy[0].data(), y), false);
                }
            };

            struct LogSoftmax : Function<LogSoftmax> {
                int64_t dim;

                auto impl_forward(TList xs) {
                    ATNN_ASSERT_EQ(xs.size(), 1);
                    auto y = at::log_softmax_forward(xs[0], this->dim);
                    this->save_for_backward({xs[0], y});
                    return y;
                }

                auto impl_backward(VList gy) {
                    ATNN_ASSERT_EQ(gy.size(), 1);
                    ATNN_ASSERT_MSG(!gy[0].train(), "this function is not double backwardable yet");
                    auto x = this->saved_tensors[0];
                    auto y = this->saved_tensors[1];
                    ATNN_ASSERT_SHAPE_EQ(gy[0].sizes(), x.sizes());
                    return Variable{at::log_softmax_backward(gy[0].data(), x, this->dim, y), false};
                }

                LogSoftmax(int64_t dim=1) : dim(dim) {}
            };

            struct Softmax : Function<Softmax> {
                int64_t dim;

                auto impl_forward(TList xs) {
                    ATNN_ASSERT_EQ(xs.size(), 1);
                    auto y = at::softmax_forward(xs[0], this->dim);
                    this->save_for_backward({xs[0], y});
                    return y;
                }

                auto impl_backward(VList gy) {
                    ATNN_ASSERT_EQ(gy.size(), 1);
                    ATNN_ASSERT_MSG(!gy[0].train(), "this function is not double backwardable yet");
                    auto x = this->saved_tensors[0];
                    auto y = this->saved_tensors[1];
                    ATNN_ASSERT_SHAPE_EQ(gy[0].sizes(), x.sizes());
                    return Variable {at::softmax_backward(gy[0].data(), x, this->dim, y), false};
                }

                Softmax(int64_t dim=1) : dim(dim) {}
            };

            struct NLLLoss : Function<NLLLoss> {
                auto impl_forward(const TList& xs) {
                    ATNN_ASSERT_EQ(xs.size(), 2);
                    this->save_for_backward(xs);
                    // if (!this->total_weight.defined()) {
                    //     this->total_weight = xs[0].type().zeros({0});
                    // } else {
                    //     this->total_weight = this->total_weight.toBackend(xs[0].type().backend());
                    // }
                    auto outs = at::nll_loss_forward(xs[0], xs[1], this->weight, this->size_average,
                                                     this->ignore_index, this->reduce);
                    this->total_weight = std::get<1>(outs);
                    return std::get<0>(outs);
                }

                auto impl_backward(const VList& gys) {
                    ATNN_ASSERT_EQ(gys.size(), 1);
                    ATNN_ASSERT_MSG(!gys[0].train(), "this function is not double backwardable yet");
                    auto xs = this->saved_tensors;
                    auto gx_data = at::nll_loss_backward(gys[0].data(), xs[0], xs[1],
                                                    this->weight, this->size_average,
                                                    this->ignore_index, this->reduce, this->total_weight);
                    return Variable {gx_data, false};
                }
                /// By default, the losses are averaged over observations for each minibatch. However, if the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True
                bool size_average = true;
                /// Specifies a target value that is ignored and does not contribute to the input gradient. When size_average is True, the loss is averaged over non-ignored targets.
                int64_t ignore_index = -1;
                /// By default, the losses are averaged or summed for each minibatch. When reduce is False, the loss function returns a loss per batch element instead and ignores size_average. Default: True
                bool reduce = true;
                at::Tensor weight, total_weight;
                NLLLoss(bool size_average=true, int64_t ignore_index=-1, bool reduce=true,
                        at::Tensor weight={}, at::Tensor total_weight={})
                    : size_average(size_average), ignore_index(ignore_index), reduce(reduce),
                        weight(weight), total_weight(total_weight)
                    {}
            };

            struct MSELoss : Function<MSELoss> {
                auto impl_forward(const TList& xs) {
                    ATNN_ASSERT_EQ(xs.size(), 2);
                    this->save_for_backward(xs);
                    return at::mse_loss_forward(xs[0], xs[1], this->size_average, this->reduce);
                }

                auto impl_backward(const VList& gy) {
                    auto xs = this->saved_tensors;

                    // TODO: support user-defined gy?
                    at::Tensor grad_out;
                    if (gy.size() == 0) { grad_out = xs[0].type().ones_like(xs[0]); }
                    else {
                        ATNN_ASSERT_EQ(gy.size(), 1);
                        grad_out = gy[0].data();
                    }
                    return Variable{at::mse_loss_backward(grad_out, xs[0], xs[1], this->size_average, this->reduce), false};
                }

                const bool size_average = true;
                const bool reduce = true;
                MSELoss(bool size_average=true, bool reduce=true)
                    : size_average(size_average), reduce(reduce)
                    {}
            };

            struct Threshold : Function<Threshold> {
                auto impl_forward(const TList& xs) {
                    ATNN_ASSERT_EQ(xs.size(), 1);
                    this->save_for_backward(xs);
                    return at::threshold_forward(xs[0], this->threshold, this->value);
                }

                auto impl_backward(const VList& gy) {
                    ATNN_ASSERT_EQ(gy.size(), 1);
                    auto x = this->saved_tensors[0];
                    ATNN_ASSERT_SHAPE_EQ(gy[0].sizes(), x.sizes());
                    return Variable{at::threshold_backward(gy[0].data(), x, this->threshold, this->value), false};
                }

                at::Scalar threshold, value;
                Threshold(double threshold, double value)
                    : threshold(threshold), value(value) {}
            };

            struct Conv2d : Function<Conv2d> {
                auto impl_forward(const TList& xs) {
                    ATNN_ASSERT_EQ(xs.size(), 3);
                    ATNN_ASSERT_EQ(xs[0].dim(), 4);
                    ATNN_ASSERT_EQ(xs[1].dim(), 4);
                    ATNN_ASSERT_EQ(xs[2].dim(), 1);

                    this->save_for_backward(xs);
                    auto&& x = xs[0];
                    auto&& w = xs[1];
                    auto&& b = xs[2];

                    at::Tensor output = x.type().zeros_like(x);
                    device::to_backend_of(this->finput, x);
                    device::to_backend_of(this->fgrad_input, x);
                    // return at::thnn_conv2d_forward_out(output, x, w, this->kernel_size, b,
                    //                                 this->stride, this->padding, this->finput, this->fgrad_input);
                    std::tie(output, this->finput, this->fgrad_input) =
                        at::thnn_conv2d_forward(x, w, this->kernel_size, b,
                                                this->stride, this->padding);
                    return output;
                }

                VList impl_backward(const VList& gys) {
                    ATNN_ASSERT_EQ(gys.size(), 1);
                    ATNN_ASSERT_MSG(!gys[0].train(), "this function is not double backwardable yet");
                    auto&& gy = gys[0].data();
                    ATNN_ASSERT_EQ(gy.dim(), 4);
                    auto&& x = this->saved_tensors[0];
                    auto&& w = this->saved_tensors[1];
                    auto&& b = this->saved_tensors[2];

                    auto gx = x.type().zeros_like(x);
                    device::to_backend_of(this->finput, x);
                    device::to_backend_of(this->fgrad_input, x);

                    at::Tensor grad_weight, grad_bias;
                    auto gw = gy.type().zeros(w.sizes());
                    auto gb = gy.type().zeros(b.sizes());
                    at::thnn_conv2d_backward_out(gx, gw, gb, gy,
                                            x, w, this->kernel_size, this->stride, this->padding,
                                            this->finput, this->fgrad_input);
                    return {Variable(gx, false), Variable(gw, false), Variable(gb, false)};
                }

                at::Tensor finput, fgrad_input; // buffers
                at::IntList kernel_size, stride, padding;

                Conv2d(int64_t out_channels, at::IntList kernel_size={3, 3},
                        at::IntList stride={1, 1}, at::IntList padding={0, 0})
                    : finput(CPU(at::kFloat).zeros(out_channels))
                    , fgrad_input(CPU(at::kFloat).zeros(out_channels))
                    , kernel_size(kernel_size)
                    , stride(stride)
                    , padding(padding) {}
            };

        } //namespace nn
    } // namespace function
} // namespace atnn
