#pragma once

#include <memory>
#include <ATen/ATen.h>

#include "traits.hpp"

namespace atnn {
    namespace memory {
        // TODO use unique_ptr if possible
        template <typename T, typename Elem>
        at::Tensor make_tensor(std::shared_ptr<T> ptr, Elem* begin, at::IntList dims) {
            constexpr auto s = traits::scalar_typeof<Elem>;
            auto deleter = [ptr](void*) mutable { ptr.reset(); };
            auto t = CPU(s).tensorFromBlob(begin, dims, deleter);
            return t;
        }

        template <typename T, typename Elem>
        at::Tensor make_tensor(std::shared_ptr<T> ptr, Elem* begin, at::IntList dims, at::IntList strides) {
            constexpr auto s = traits::scalar_typeof<Elem>;
            auto deleter = [ptr](void*) mutable { ptr.reset(); };
            auto t = CPU(s).tensorFromBlob(begin, dims, strides, deleter);
            return t;
        }

        /// NOTE this function takes ownership from Elem* begin and delete when shared_ptr is reset
        template <typename Elem>
        at::Tensor make_tensor(Elem* begin, at::IntList dims) {
            return make_tensor(std::shared_ptr<Elem>(begin), dims);
        }

        /// NOTE this function takes ownership from Elem* begin and delete when shared_ptr is reset
        template <typename Elem>
        at::Tensor make_tensor(Elem* begin, at::IntList dims, at::IntList strides) {
            return make_tensor(std::shared_ptr<Elem>(begin), dims, strides);
        }


        template <typename Elem>
        at::Tensor make_tensor(std::shared_ptr<std::vector<Elem>> ptr) {
            return make_tensor(ptr, ptr->data(), ptr->size());
        }

        template <typename Elem>
        at::Tensor make_tensor(std::shared_ptr<std::vector<Elem>> ptr, at::IntList dims) {
            return make_tensor(ptr, ptr->data(), dims);
        }

#ifdef __has_include
#if __has_include(<kaldi-matrix.h>)
#include <kaldi-matrix.h>
        template <typename Elem>
        at::Tensor make_tensor(std::shared_ptr<kaldi::Vector<Elem>> ptr) {
            return make_tensor(ptr, ptr->Data(), ptr->Dim());
        }

        template <typename Elem>
        at::Tensor make_tensor(std::shared_ptr<kaldi::Matrix<Elem>> ptr) {
            return make_tensor(ptr, ptr->Data(), {ptr->NumRows(), ptr->NumCols()}, {ptr->Stride(), 1});
        }
#endif
#endif
    }
}
