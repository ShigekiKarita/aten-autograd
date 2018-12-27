#include <iostream>
#include <memory>

#include <kaldi-io.h>
#include <kaldi-table.h>
#include <kaldi-matrix.h>
#include <table-types.h>

#include <ATen/ATen.h>
#include <atnn/variable.hpp>
#include <atnn/memory.hpp>
#include <atnn/testing.hpp>

int main(int argc, char**argv) {
    atnn::test_common(argc, argv, []([[gnu::unused]] auto device) {

    // testing float-vector io
    auto fvec_scp = "scp:data/conf.scp";
    auto fvec_scp_out = "ark,t:data/conf_out.ark"; // "t" is text

    // see http://kaldi-asr.org/doc/io.html
    kaldi::SequentialBaseFloatVectorReader feature_reader(fvec_scp);
    kaldi::BaseFloatVectorWriter feature_writer(fvec_scp_out);
    for(; !feature_reader.Done(); feature_reader.Next()) {
        std::string utt = feature_reader.Key();

        // see http://kaldi-asr.org/doc/matrix.html
        // http://kaldi-asr.org/doc/classkaldi_1_1VectorBase.html
        auto new_feats = std::make_shared<kaldi::Vector<kaldi::BaseFloat>>(feature_reader.Value());
        auto t = atnn::memory::make_tensor(new_feats);

        ATNN_ASSERT_EQ(t.dim(), 1);
        ATNN_ASSERT_EQ(new_feats->Dim(), t.size(0));

        t[0] = 23;
        // std::cout << *new_feats << std::endl;
        // TEST overwritten by at::Tensor t
        ATNN_ASSERT_EQ(*(new_feats->Data()), 23);
        feature_writer.Write(utt, *new_feats);
    }

    {
        auto m = std::make_shared<kaldi::Matrix<kaldi::BaseFloat>>(3, 4);
        m->SetRandn();
        auto t = atnn::memory::make_tensor(m);
        t[0][1] = 23;
        ATNN_ASSERT_EQ(m->Index(0, 1), 23);
    }
    {
        auto m = std::make_shared<kaldi::Matrix<kaldi::BaseFloat>>(3, 4, kaldi::kSetZero, kaldi::kStrideEqualNumCols);
        m->SetRandn();
        auto t = atnn::memory::make_tensor(m);
        t[0][1] = 23;
        ATNN_ASSERT_EQ(m->Index(0, 1), 23);
    }


    }, true);
}
