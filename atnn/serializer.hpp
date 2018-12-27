/**
  \file serializer.hpp
  \brief serialization utility
  \example test_serializer.cpp HDF5 serialization example
 */

#pragma once
#include <fstream>
#include <memory>
#include <unordered_map>

#include <H5Cpp.h>
#include <ATen/ATen.h>
#include "testing.hpp"

namespace atnn {
    /**
        \brief serialization utility
        \ingroup atnn::serializer
    */
    namespace serializer {
        /**
           ATen ScalarType to H5::PredType conversion dict.

           \ref https://support.hdfgroup.org/HDF5/doc/cpplus_RM/class_h5_1_1_pred_type.html
         */
        static const std::unordered_map<at::ScalarType, H5::PredType> type_dict = {
            {at::kChar, H5::PredType::NATIVE_INT8},
            {at::kByte, H5::PredType::NATIVE_UINT8},
            {at::kFloat, H5::PredType::NATIVE_FLOAT},
            {at::kDouble, H5::PredType::NATIVE_DOUBLE},
            {at::kShort, H5::PredType::NATIVE_INT16},
            {at::kInt, H5::PredType::NATIVE_INT}, // ??
            {at::kLong, H5::PredType::NATIVE_INT64}
            /// \todo add half float
        };

        using H5FAccType = unsigned int; // decltype(H5F_ACC_TRUNC); macro error

        /// H5check() in H5F_ACC_XXX macro prevents constatnt integer initialization
        enum class H5FileAccess : H5FAccType {
            /// Open file as read-only, if it already exists, and fail, otherwise
            read_only = 0x0000u, // H5F_ACC_RDONLY,
            /// Open file for read/write, if it already exists, and fail, otherwise
            read_write = 0x0001u, // H5F_ACC_RDWR,
            /// Truncate file, if it already exists, erasing all data previously stored in the file.
            truncate =  0x0002u, // H5F_ACC_TRUNC,
            /// Fail if file already exists. H5F_ACC_TRUNC and H5F_ACC_EXCL are mutually exclusive
            exclusive = 0x0004u, // H5F_ACC_EXCL,
            // Append  file (i.e., open file for read write), if it already exisits, and truncate file, otherwise
            append
        };

        struct H5ShapeChangedError : std::runtime_error {
            H5ShapeChangedError(const std::string& msg) : std::runtime_error(msg) {}
        };

        bool file_exists(const std::string& path) {
            return std::ifstream(path).is_open();
        }

        struct H5Serializer {
            // https://support.hdfgroup.org/HDF5/doc/cpplus_RM/class_h5_1_1_pred_type.html
            std::string path;
            H5FileAccess access;

            H5Serializer(const std::string& path, H5FileAccess access=H5FileAccess::append)
                : path(path), access(access) {}

            auto file_ptr() const {
                H5FileAccess actual_access = this->access;;
                if (this->access == H5FileAccess::append) {
                    actual_access = file_exists(this->path)
                        ? H5FileAccess::read_write
                        : H5FileAccess::truncate;
                }
                return std::make_unique<H5::H5File>(this->path, static_cast<H5FAccType>(actual_access));
            }

            /**
               HDF5 Tensor reader/writer

               \todo support "/" separated keys e.g., "aaa/bbb/ccc" (see. H5roup)
               \ref https://support.hdfgroup.org/HDF5/doc/cpplus_RM/writedata_8cpp-example.html
             */
            struct Accessor {
                const H5Serializer* h5_ptr;
                std::string key;

                Accessor(const H5Serializer* ptr, const std::string& key)
                    : h5_ptr(ptr), key(key) {}

                static auto dataspace_shape(const H5::DataSpace& fspace) {
                    std::vector<hsize_t> hsizes(fspace.getSimpleExtentNdims());
                    fspace.getSimpleExtentDims(hsizes.data());
                    std::vector<int64_t> tsizes;
                    tsizes.reserve(hsizes.size());
                    std::copy(hsizes.begin(), hsizes.end(), std::back_inserter(tsizes));
                    return tsizes;
                }

                /// implementation of at::Tensor = h5dict["key"];
                operator at::Tensor() const {
                    // close by RAII
                    auto dataset = std::make_unique<H5::DataSet>(this->h5_ptr->file_ptr()->openDataSet(this->key));
                    auto tsizes = dataspace_shape(dataset->getSpace());
                    at::Tensor ret;
                    auto pred_type = dataset->getDataType();
                    /// \todo use boost::bimap
                    for (auto type : type_dict) {
                        if (type.second == pred_type) {
                            ret = at::CPU(type.first).zeros(tsizes);
                        }
                    }
                    dataset->read(ret.data_ptr(), pred_type);
                    return ret;
                }

                /// implementation of h5dict["key"] = tensor;
                auto& operator=(const at::Tensor& t) {
                    const auto pred_type = type_dict.at(t.type().scalarType());
                    int fillvalue = 0;   /* Fill value for the dataset */
                    H5::DSetCreatPropList plist;
                    plist.setFillValue(pred_type, &fillvalue);

                    auto tsizes = t.sizes();
                    std::vector<hsize_t> hsizes;
                    hsizes.reserve(tsizes.size());
                    std::copy(tsizes.begin(), tsizes.end(), std::back_inserter(hsizes));
                    H5::DataSpace fspace(t.dim(), hsizes.data());

                    auto fp = this->h5_ptr->file_ptr();
                    std::unique_ptr<H5::DataSet> dataset;
                    try { // FIXME: do something like if (fp->datasetExists(this->key)) {
                        H5::Exception::dontPrint();
                        dataset = std::make_unique<H5::DataSet>(fp->openDataSet(this->key));
                        if (!atnn::shape_eq(t.sizes(), dataspace_shape(dataset->getSpace()))) {
                            // \todo: display sizes
                            atnn::throw_with_trace(
                                H5ShapeChangedError("shape change (HDF5 does not support deletion or shape-changed overwrite)"));
                        }
                    } catch(H5::FileIException&) {
                        // file not found
                        dataset = std::make_unique<H5::DataSet>(fp->createDataSet(this->key, pred_type, fspace, plist));
                    } catch(H5::GroupIException&) {
                        // group not found
                        dataset = std::make_unique<H5::DataSet>(fp->createDataSet(this->key, pred_type, fspace, plist));
                    }
                    dataset->write(t.toBackend(at::kCPU).contiguous().data_ptr(), pred_type);
                    return *this;
                }
            };

            auto& dump(const std::unordered_map<std::string, at::Tensor>& dict) {
                for (const auto& item : dict) {
                    Accessor(this, item.first) = item.second;
                }
                return *this;
            }

            auto load() const {
                std::unordered_map<std::string, at::Tensor> ret;
                auto fp = this->file_ptr();
                for (hsize_t hid = 0; hid < fp->getNumObjs(); ++hid) {
                    auto name = fp->getObjnameByIdx(hid);
                    ret[name] = Accessor(this, name);
                }
                return ret;
            }

            Accessor operator[](const std::string& key) {
                return {this, key};
            }
        };
    } // namespace serializer
} // namespace atnn
