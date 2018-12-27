#pragma once
#include <ATen/ScalarType.h>

namespace atnn {
    namespace traits {
        template <typename T>
        struct ScalarTypeof;

        using at::Half;

#define ATNN_SCALAR_TYPE_OF(_1,n,_2) \
        template <> struct ScalarTypeof<_1> { constexpr static at::ScalarType value = at::k##n ; };
        AT_FORALL_SCALAR_TYPES(ATNN_SCALAR_TYPE_OF)
#undef ATNN_SCALAR_TYPE_OF

        template <typename T>
        constexpr at::ScalarType scalar_typeof = ScalarTypeof<T>::value;
    }
}
