/**
   \file variable.hpp
   \brief Autograd variable as a computation graph node
   \example test_variable.cpp
*/
#pragma once

#include <algorithm>
#include <numeric> // accumulate
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <type_traits>
#include <functional>

#include <ATen/ATen.h>

#include "traits.hpp"
#include "testing.hpp"

/**
 * \brief A tensor and neural-network library's global namespace
 *
 * \ingroup atnn
 */
namespace atnn {
    namespace detail {
        template <typename E>
        auto nested_init_list_shape(const E&, std::vector<int64_t>&& shape) ->
            std::enable_if_t<std::is_fundamental<E>::value, decltype(shape)> {
            return std::move(shape);
        }

        template <typename E>
        auto nested_init_list_shape(const std::initializer_list<E>& il,
                                    std::vector<int64_t>&& shape={}) {
            shape.push_back(il.size());
            return nested_init_list_shape(*il.begin(), std::move(shape));
        }

        template <typename E>
        constexpr bool is_element = std::is_fundamental<E>::value;

        template <typename E>
        constexpr std::enable_if_t<is_element<E>, E> deep_elem(E) {
            return E{};
        }

        template <typename Container, typename _ = std::enable_if_t<!is_element<Container>>>
        constexpr auto deep_elem(Container il) {
            return deep_elem(*il.begin());
        }

        template <typename E>
        using DeepElementTypeof = std::remove_cv_t<decltype(deep_elem(std::declval<E>()))>;

        template <typename E>
        std::enable_if_t<is_element<E>> flatten_list(E e, std::vector<E>& dst) {
            dst.push_back(e);
        }

        template <typename T>
        void flatten_list(std::initializer_list<T> xs, std::vector<DeepElementTypeof<T>>& dst) {
            if (dst.capacity() == 0) {
                auto shape = nested_init_list_shape(xs);
                auto length = std::accumulate(std::begin(shape), std::end(shape), 1,
                                              std::multiplies<>());
                dst.reserve(length);
            }
            for (auto x: xs) {
                flatten_list(x, dst);
            }
        }

        template <typename T, size_t N>
        struct _IL {
            using type = std::initializer_list<typename _IL<T, N-1>::type>;
        };

        template <typename T>
        struct _IL<T, 1> {
            using type = std::initializer_list<T>;
        };
    } // namespace detail

    template <typename T, size_t N=1>
    using Init = typename detail::_IL<T, N>::type;

    template <typename T>
    using IL = std::initializer_list<T>;

    struct Variable; // forward declaration for FunctionBase
    Variable operator+(const Variable& lhs, const Variable& rhs); // need for Variable::backward
    struct VariableSlice; // : Variable;
    using VList = std::vector<Variable>;
    using TList = std::vector<at::Tensor>;

    /**
      Base class of atnn::function::Function<Derived>
     */
    struct FunctionBase : std::enable_shared_from_this<FunctionBase> {
        TList saved_tensors;
        VList saved_variables;
        std::vector<at::IntList> saved_sizes;
        VList vargs, vrets; /// \todo use unordered_set intead of vector

        virtual ~FunctionBase() {}
        // virtual TList backward(TList grads) = 0;
        virtual void toBackend(at::Backend b) = 0;
        virtual void register_grad_output(Variable, const Variable&) = 0;
        virtual VList compute_grad_inputs() = 0;
    };

    using FunctionPtr = std::shared_ptr<FunctionBase>;

    /**
       VariableStorage is just a storage for data/grad in Variable
       and to be managed via std::shared_ptr
     */
    struct VariableStorage {
        std::unique_ptr<at::Tensor> data;
        std::shared_ptr<Variable> grad;
        bool train;
        VariableStorage(at::Tensor data, bool train=true) : data(std::make_unique<at::Tensor>(data)), train(train) {}
    };

    /**
       Variable is a autograd tensor.
       This class does not consist any autograd Funcion or Module.
       Operator(+,*,-,/) overloads are defined in atnn namespace at function.hpp
     */
    struct Variable {
        std::shared_ptr<VariableStorage> ptr;
        FunctionPtr function;

        struct Hash {
            size_t operator()(const Variable& v) const {
                return reinterpret_cast<size_t>(v.ptr.get()) / sizeof(VariableStorage);
            }
        };

        struct Equal {
            bool operator()(const Variable& a, const Variable& b) const {
                return b.ptr.get() == a.ptr.get();
            }
        };

        bool operator==(const Variable& that) const {
            return Equal()(*this, that);
        }

        bool operator!=(const Variable& that) const {
            return !Equal()(*this, that);
        }

        using Set = std::unordered_set<Variable, Hash, Equal>;

        template <typename Value>
        using Map = std::unordered_map<Variable, Value, Hash, Equal>;

        Variable() : ptr(std::make_shared<VariableStorage>(at::Tensor{}, true)) {}

        Variable(at::Tensor data, bool train=true)
            : ptr(std::make_shared<VariableStorage>(data, train)) {}

        template <typename T>
        static auto make_storage_ptr(IL<T> blob, at::Backend backend = at::kCPU, bool train=true) {
            using E = detail::DeepElementTypeof<T>;
            auto shape = detail::nested_init_list_shape(blob);
            auto vec_ptr = std::make_shared<std::vector<E>>();
            detail::flatten_list(blob, *vec_ptr);
            auto deleter = [vec_ptr](void*) mutable { vec_ptr.reset(); };
            auto tensor = at::getType(backend, traits::scalar_typeof<E>).tensorFromBlob(vec_ptr->data(), shape, deleter);
            return std::make_shared<VariableStorage>(tensor, train);
        }

        // very very redundant but necessary to support something like Variable v = {{1,2}, {3,4}}
        template <typename T>
        Variable(IL<T> blob, at::Backend backend = at::kCPU, bool train=true)
            : ptr(make_storage_ptr(blob, backend, train)) {}
        template <typename T>
        Variable(IL<IL<T>> blob, at::Backend backend = at::kCPU, bool train=true)
            : ptr(make_storage_ptr(blob, backend, train)) {}
        template <typename T>
        Variable(IL<IL<IL<T>>> blob, at::Backend backend = at::kCPU, bool train=true)
            : ptr(make_storage_ptr(blob, at::kCPU, true)) {}
        template <typename T>
        Variable(IL<IL<IL<IL<T>>>> blob, at::Backend backend = at::kCPU, bool train=true)
            : ptr(make_storage_ptr(blob, backend, train)) {}
        template <typename T, size_t N>
        Variable(Init<T, N> blob, at::Backend backend = at::kCPU, bool train=true)
            : ptr(make_storage_ptr(blob, backend, train)) {}

        auto& train() {
            return this->ptr->train;
        }

        auto train() const {
            return this->ptr->train;
        }

        const auto& data() const {
            return *(this->ptr->data);
        }

        auto& data() {
            return *(this->ptr->data);
        }

        /**
            generated `auto method(void)` in at::Tensor
         */
#define ATNN_TENSOR_NOARG_METHODS(_) \
        _(defined) \
        _(dim) \
        _(sizes) \
        _(numel) \
        _(data_ptr) \
        _(is_signed) \
        _(get_device) \
        _(is_contiguous) \
        _(storage_offset) \
        _(all) \
        _(any)

#define ATNN_DEFINE_NOARG_METHODS(NAME) \
        auto NAME() const { return this->ptr->data->NAME(); }

        ATNN_TENSOR_NOARG_METHODS(ATNN_DEFINE_NOARG_METHODS)
#undef ATNN_TENSOR_NOARG_METHODS
#undef ATNN_DEFINE_NOARG_METHODS

        bool is_same_size(const Variable& v) const {
            return this->data().is_same_size(v.data());
        }

        auto size(int64_t dim) const {
            return this->data().size(dim);
        }
        // const auto& contiguous() const {
        //     this->data().swap(this->data().contiguous());
        //     return *this;
        // }

        auto contiguous() const {
            if (this->data().is_contiguous()) return *this;
            return Variable(this->data().contiguous(), this->train());
        }

        auto grad() const {
            return *(this->ptr->grad);
        }

        void clear_grads() {
            // if (!this->ptr->grad) return;

            this->ptr->grad.reset();
            if (!this->is_leaf()) {
                for (auto& v: this->children()) {
                    v.clear_grads();
                }
            }
        }

        void detach() {
            if (!this->function) return;
            // should i detach all?
            if (!this->is_leaf()) {
                for (auto& v: this->children()) {
                    v.detach();
                }
            }
            this->function.reset();
        }

        auto& set_function(FunctionPtr m) {
            if (this->train()) {
                this->function = m;
            }
            return *this;
        }

        bool is_leaf() const { return this->function == nullptr; }

        VList& children() { return this->function->vargs; }

        void backward(const at::Tensor& grad) {
            this->backward(Variable(grad, false));
        }

        void backward(const Variable& grad) {
            if (!this->is_leaf() && this->function->vrets.empty()) return; //???

            // ATNN_ASSERT_SHAPE_EQ(this->sizes(), grad.sizes());
            if (this->ptr->grad && this->ptr->grad->defined()) {
                this->ptr->grad = std::make_shared<Variable>(*(this->ptr->grad) + grad);
            } else {
                this->ptr->grad = std::make_shared<Variable>(grad);
            }

            if (this->is_leaf()) return; // stop the recursion

            this->function->register_grad_output(*this, *(this->ptr->grad));
            size_t i = 0;
            for (auto&& g: this->function->compute_grad_inputs()) {
                this->children()[i].backward(g);
                ++i;
            }
            this->function.reset();
        }

        void backward() {
            ATNN_ASSERT(this->train());
            // ATNN_ASSERT_SHAPE_EQ(this->sizes(), at::IntList {1});
            this->backward(this->data().type().ones({1}));
        }

        void toBackend(at::Backend b) {
            if (this->defined()) this->ptr->data = std::make_unique<at::Tensor>(this->data().toBackend(b));
            if (this->ptr->grad) this->ptr->grad->toBackend(b);
        }

        /**
          autograd functions implemented in chain.hpp
          \todo implement operator+,-,*,/ (at::Scalar)
          \todo implement operator+=,-=,*=,/=(Variable, at::Scalar)
         */
        Variable select(int64_t dim, int64_t index) const;
        Variable operator[](int64_t index) const { return this->select(0, index); };
        Variable index_select(int64_t dim, at::Tensor index) const;
        Variable operator[](at::Tensor index) const { return this->index_select(0, index); }
        Variable view(at::IntList sizes, const char* debug="") const;
        Variable transpose(int64_t dim1, int64_t dim2) const;
        Variable permute(at::IntList dims) const;
        Variable t() const { return this->transpose(0, 1); }
        Variable log() const;
        Variable pow(double e) const;
        Variable pow(Variable e) const;
        Variable sqrt() const { return this->pow(0.5); }
        Variable sum(int64_t dim, bool keepdim=false) const;
        Variable sum() const;
        Variable slice(int64_t dim=0, int64_t start=0, int64_t end=9223372036854775807, int64_t step=1) const;

    private:
        Variable slice_dim_first(int64_t dim, int64_t start=0, int64_t end=9223372036854775807, int64_t step=1) {
            /// \note this 0-to-default is for operator[](const Index(&args)[N][3]) because int a[3] = {1} set 0 to a[1] and a[2]
            if (end == 0) { end=9223372036854775807; }
            if (step == 0) { step = 1; }
            return this->slice(dim, start, end, step);
        }

        template <typename Index=int, size_t N, size_t ... I>
        Variable variadic_slice(const Index(&args)[N], std::index_sequence<I...>) {
            return this->slice(args[I]...);
        }

        template <typename Index=int, size_t ... I>
        Variable variadic_slice_dim(size_t dim, const Index(&args)[3], std::index_sequence<I...>) {
            return this->slice_dim_first(dim, args[I]...);
        }

    public:
        template <typename Index=int, size_t N>
        Variable operator[](const Index(&args)[N]) {
            static_assert(N <= 4, "[int64_t args[N]] should be [{int64_t start, end, step, dim}]");
            return this->variadic_slice(args, std::make_index_sequence<N>{});
        }

        template <typename Index=int, size_t N>
        Variable operator[](const Index(&args)[N][3]) {
            Variable ret = *this;
            for (size_t n = 0; n < N; ++n) {
                ret = ret.variadic_slice_dim(n, args[n], std::make_index_sequence<3>{});
            }
            return ret;
        }

        /// \brief Returns a new tensor with a dimension of size one inserted at the specified position.
        Variable unsqueeze(int64_t dim) const {
            if (dim < 0) {
                dim += this->dim() + 1;
            }
            ATNN_ASSERT(0 <= dim && dim <= this->dim());
            auto n = this->dim() + 1;
            std::vector<int64_t> s;
            s.reserve(n);
            for (int64_t d = 0; d < n; ++d) {
                int64_t i = 1;
                if (d < dim) {
                    i = this->size(d);
                } else if (d > dim) {
                    i = this->size(d - 1);
                }
                s.push_back(i);
            }
            return this->view(s);
        }

        Variable squeeze() const {
            std::vector<int64_t> v = this->sizes();
            auto result = std::remove_if(v.begin(), v.end(), [](int64_t x) { return x == 1; });
            v.erase(result, v.end());
            return this->view(v);
        }

        Variable squeeze(int64_t dim) const {
            if (dim < 0) {
                dim += this->dim();
            }
            ATNN_ASSERT(0 <= dim && dim < this->dim());
            std::vector<int64_t> v = this->sizes();
            ATNN_ASSERT(v[dim] == 1);
            v.erase(v.begin() + dim);
            return this->view(v);
        }

        /**
         * \brief Splits the tensor into equally sized chunks (if possible).
         * Last chunk will be smaller if the tensor size aint64_t a given dimension is not divisible by split_size.
         */
        VList split(int64_t split_size, int64_t dim);

        /**
         * \brief Splits a tensor into a number of chunks aint64_t a given dimension.
         * A number of chunks and last chunk can be smaller than specified.
         */
        VList chunk(int64_t chunks, int64_t dim) {
            return this->split((this->size(dim) + chunks - 1) / chunks, dim);
        }
    };


    bool all_train(const VList& vs) noexcept {
        return std::all_of(vs.begin(), vs.end(), [](auto&& v) { return v.train(); });
    }


    std::ostream& operator<<(std::ostream &strm, const Variable &v) {
        return strm << "Variable(\n"
                    << "data=\n" << v.data()
                    // << "\ngrad=\n" << v.grad()
                    << "\n)";
    }

} // namespace atnn
