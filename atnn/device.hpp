/**
    \file device.hpp
    \brief Memory allocation device (CPU/CUDA) utility
*/
#pragma once

namespace atnn {
    /**
        \brief Memory allocation device (CPU/CUDA) utility
        \ingroup atnn::device
    */
    namespace device {
        template <typename T1, typename T2>
        static void to_backend_of(T1& src, const T2& dst) {
            const auto src_backend = src.type().backend();
            const auto dst_backend = dst.type().backend();
            if (src_backend != dst_backend) {
                src = src.toBackend(dst_backend);
            }
        }

        template <typename T>
        static void to_backend_of(T& src, at::Backend b) {
            const auto src_backend = src.type().backend();
            if (src_backend != b) {
                src = src.toBackend(b);
            }
        }
    }
}
