#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>

#include <ATen/ATen.h>
#include <atnn/chain.hpp>
#include <atnn/optimizer.hpp>
#include <atnn/dataset.hpp>
#include <atnn/dataloader.hpp>
#include <atnn/serializer.hpp>
#include <atnn/testing.hpp>
#include <atnn/memory.hpp>


namespace C = atnn::chain;
namespace nn = atnn::chain::nn;
namespace F = atnn::function;
namespace D = atnn::dataset;
namespace S = atnn::sampler;
namespace L = atnn::dataloader;
namespace O = atnn::optimizer;


auto load_mnist(const std::string& dir) {
    std::unordered_map<std::string, std::unordered_map<std::string, at::Tensor>> dict;
    for (std::string&& filename : {"t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", "train-images.idx3-ubyte", "train-labels.idx1-ubyte"}) {
        std::ifstream file(dir + "/" + filename, std::ios::binary);
        auto is_train = filename.find("train") != std::string::npos;
        auto is_img = filename.find("images") != std::string::npos;

        file.seekg(is_img ? 16 : 8);
        auto bytes = std::make_shared<std::vector<unsigned char>>(std::istreambuf_iterator<char>(file),
                                                                  std::istreambuf_iterator<char>());
        long n_samples = static_cast<long>(bytes->size())  / (is_img ? (28 * 28) : 1);
        std::cout << filename << " loaded: " << n_samples << std::endl;
        if (is_img) {
            dict[is_train ? "train" : "test"]["input"] = atnn::memory::make_tensor(bytes, {n_samples, 28, 28});
        } else {
            dict[is_train ? "train" : "test"]["target"] = atnn::memory::make_tensor(bytes);
        }
    }
    return dict;
}


#define PARAM(name) {#name, (name)}


struct Net : C::Chain {
    C::ChainPtr<nn::Linear> linear1, linear2, linear3;

    Net(long n_units=1000)
        : linear1(new nn::Linear(28*28, n_units))
        , linear2(new nn::Linear(n_units, n_units))
        , linear3(new nn::Linear(n_units, 10)) {
        this->chain_dict = {
            PARAM(linear1),
            PARAM(linear2),
            PARAM(linear3)
        };
    }

    auto operator()(atnn::Variable x) {
        auto y1 = nn::relu(this->linear1(x));
        auto y2 = nn::relu(this->linear2(y1));
        return this->linear3(y2);
    }
};


int main(int argc, char** argv) {
    atnn::test_common(argc, argv, [](auto device) {
            if (at::hasCUDA() && device == at::CPU) return;
            auto n_batch = 100;
            auto tensor_dict = load_mnist("./data/");
            if (!at::hasCUDA()) {
                int64_t max_size = 10000;
                tensor_dict["train"]["input"].resize_({max_size, 28, 28});
                tensor_dict["train"]["target"].resize_({max_size});
            }

            auto train_set = std::make_shared<D::TensorDataset>(tensor_dict["train"]);
            auto train_loader = L::DataLoader<S::RandomSampler>(train_set, n_batch);
            auto test_set = std::make_shared<D::TensorDataset>(tensor_dict["test"]);
            auto test_loader = L::DataLoader<S::RandomSampler>(test_set, n_batch);

            Net net;
            if (device == at::CUDA) {
                net.toBackend(at::kCUDA);
            }
            auto optimizer = O::Adam(net.parameters());
            for (long epoch = 0; epoch < 3; ++epoch) {
                for (bool train : {true, false}) {
                    double sum_loss = 0.0;
                    long sum_acc = 0;
                    net.set_train(train);
                    auto& loader = train ? train_loader : test_loader;
                    for (auto batch : loader) {
                        atnn::Variable xs(batch["input"].toType(at::kFloat).view({-1, 28 * 28}) / 255.0, train);
                        atnn::Variable ts(batch["target"].toType(at::kLong), train);
                        if (device == at::CUDA) {
                            xs.toBackend(at::kCUDA);
                            ts.toBackend(at::kCUDA);
                        }
                        auto ys = net(xs);
                        auto loss = nn::cross_entropy(ys, ts);
                        auto loss_val = at::Scalar(loss.data()).toDouble();
                        sum_loss += loss_val * ys.data().size(0);
                        auto index = std::get<1>(ys.data().max(1)); // [value, index]
                        auto acc_val = at::Scalar(index.eq(ts.data()).toType(at::kLong).sum()).toLong();
                        sum_acc += acc_val;
                        if (train) {
                            loss.clear_grads();
                            loss.backward();
                            optimizer.update();
                            if (!at::hasCUDA()) {
                                std::cout << "loss: " << loss_val << ", acc: " << acc_val << std::endl;
                            }
                        }
                    }
                    auto n_sample = loader.dataset->size();
                    std::cout << (train ? "train" : "test ") << " epoch: " << epoch
                              << ",\tloss: " << sum_loss / n_sample
                              << ",\tacc: " << static_cast<double>(sum_acc) / n_sample << std::endl;
                }
                atnn::serializer::H5Serializer h5("/tmp/mnist.hdf5");
                h5.dump(net.state_dict());
            }
        });
}
