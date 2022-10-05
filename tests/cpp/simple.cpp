#include <wavelet.h>

#include <vector>

int main()
{
    // define a 1-dimensional volume of size 128
    std::vector<float> input(128);
    for (std::size_t i = 0; i < input.size(); i++) {
        input[i] = static_cast<float>(i);
    }

    // dimmensions of the input, must be greader than num_base_dims
    constexpr std::size_t num_base_dims = 1;
    std::array<std::size_t, num_base_dims + 1> dims { 128, 1 };
    std::array<std::uint32_t, num_base_dims + 1> levels { 7, 0 };
    std::array<wavelet::range<std::size_t>, num_base_dims + 1> roi {
        wavelet::range<std::size_t> { 0, dims[0] },
        wavelet::range<std::size_t> { 0, dims[1] }
    };

    // construct the encoder with the dimensions of the dataset.
    wavelet::slice<std::size_t> dims_slice { dims };
    wavelet::encoder<float> enc { dims_slice, num_base_dims };

    // the fetcher is a simple lambda, reading a value out of the vector.
    // the vector is only read, therefore the lambda is thread-safe.
    wavelet::volume_fetcher<float> fetcher {
        [&input](wavelet::slice<const std::size_t> pos) { return input[pos[0]]; }
    };

    // insert the constructed fetcher into the encoder at position [0].
    enc.add_fetcher(wavelet::slice<const std::size_t> { 0 }, std::move(fetcher));

    // write some example metadata.
    constexpr const char* example_string = "some example string";
    enc.metadata_insert("example float", 1.0f);
    enc.metadata_insert("example string", wavelet::string { example_string });

    auto i_meta = enc.metadata_get<int>("invalid key");
    auto f_meta = enc.metadata_get<float>("example float");

    assert(!i_meta.has_value());
    assert(f_meta.has_value() && *f_meta == 1.0f);

    // encode the data with a block size of 32x1 and the average filter.
    std::array<std::size_t, num_base_dims + 1> block_size { 32, 1 };
    wavelet::slice<std::size_t> block_size_slice { block_size };
    enc.encode<wavelet::filters::average_filter>("encode_cpp", block_size_slice);

    // construct a decoder by opening the newly encoded file.
    wavelet::decoder<float, wavelet::filters::average_filter> dec { "encode_cpp/output.bin" };

    // metadata added to the encoder are available to the decoder.
    auto i_meta_dec = dec.metadata_get<int>("invalid key");
    auto f_meta_dec = dec.metadata_get<float>("example float");
    auto str_meta_dec = dec.metadata_get<wavelet::string>("example string");

    assert(!i_meta_dec.has_value());
    assert(f_meta_dec.has_value() && *f_meta_dec == 1.0f);
    assert(str_meta_dec.has_value());

    std::string str { str_meta_dec->begin(), str_meta_dec->end() };
    assert(str == example_string);

    std::vector<float> output(128);

    // the writer splits up the output into the number of blocks,
    // requested by the decoder and returns another callable.
    // the writer will only be called once and does not need to be
    // thread-safe.
    wavelet::writer_fetcher<float> writer {
        [&output](wavelet::slice<const std::size_t> block_counts) -> wavelet::block_writer_fetcher<float> {
            // the volume is one dimensional, therefore we
            // only require the offset to the start of the
            // requested block.
            auto block_size = 128 / block_counts[0];
            std::vector<std::size_t> block_offsets(block_counts[0]);
            for (std::size_t i = 0; i < block_counts[0]; i++) {
                block_offsets[i] = i * block_size;
            }

            // the returned callable uniquely assigns a block to each decoding task.
            // all captures originating from the writer_fetcher must be captured by
            // value. must be thread-safe and callable multiple times.
            return wavelet::block_writer_fetcher<float> {
                [block_offsets, &output](std::size_t block) -> wavelet::block_writer<float> {
                    auto block_offset = block_offsets[block];
                    return wavelet::block_writer<float> {
                        [block_offset, &output](wavelet::slice<const std::size_t> pos, float value) {
                            output[pos[0] + block_offset] = value;
                        }
                    };
                }
            };
        }
    };

    // decode the entire dataset.
    wavelet::slice<std::uint32_t> levels_slice { levels.data(), levels.size() };
    wavelet::slice<wavelet::range<std::size_t>> roi_slice { roi.data(), roi.size() };
    dec.decode(std::move(writer), roi_slice, levels_slice);

    assert(output == input);

    return 0;
}