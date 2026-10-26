// Images from a request to the vision tower's input (specs/DREAMFERENCE_LING_ENGINE_VISION.md §20.1, §20.3 (c), §20.6):
// the data: URL, the decoders (format sniffed from the bytes), production's preprocessing (smart_resize, PyTorch's
// antialiased bicubic as its kernel computes it, the normalisation, the merge-window patch order), the image's hash
// and its id in the prompt (§20.7). Runs on the HTTP handler's thread, never the engine's.
#pragma once

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ling::serve {

// A request error about an image: the handler answers 400 with this message and code.
struct ImageError : std::invalid_argument {
  std::string code;
  ImageError(const std::string& msg, std::string c) : std::invalid_argument(msg), code(std::move(c)) {}
};

struct DecodedImage {
  int width = 0, height = 0;
  std::vector<uint8_t> rgb;  // [height][width][3]
};

// The bytes of a `data:<mime>;base64,<payload>` URL or of bare base64. http(s), file:// and paths: ImageError
// (image_url_not_supported). Payloads above max_bytes decoded: ImageError (image_too_large).
std::string image_bytes_from_url(const std::string& url, size_t max_bytes = 20u << 20);

enum class ImageFormat { Unknown, Png, Jpeg, Webp, Gif, Bmp, Tiff, Heif };
ImageFormat sniff_image_format(const std::string& bytes);
const char* image_format_name(ImageFormat f);

// How JPEGs are decoded: nvJPEG on the GPU as production does (§20.2 item 2), falling back to libjpeg-turbo when
// nvJPEG refuses a file; or libjpeg-turbo only.
enum class JpegDecoder { Gpu, Cpu };
void set_jpeg_decoder(JpegDecoder d);

// Decodes PNG, JPEG or WebP to 8-bit RGB as production's loader leaves it (alpha dropped, not composited; no EXIF
// rotation). Anything else: ImageError (unsupported_image_format); a corrupt file: ImageError (invalid_image).
DecodedImage decode_image(const std::string& bytes);
DecodedImage decode_jpeg_cpu(const std::string& bytes);

// The processor's settings (preprocessor_config.json).
struct ProcessorConfig {
  int patch = 16, merge = 2, temporal = 2;
  long min_pixels = 65536, max_pixels = 16777216;
  int factor() const { return patch * merge; }
};

// transformers 5.12.1's smart_resize (Python's round: halves to even). Aspect ratio above 200: ImageError.
void smart_resize(int height, int width, int factor, long min_pixels, long max_pixels, int& out_h, int& out_w);

// torchvision's resize(antialias=True, BICUBIC) of a uint8 image on the GPU: FP32, PyTorch's separable antialiased
// bicubic (a = -0.5) as upsample_gen2d_aa computes it, clamp to [0, 255], round half to even. In: [h][w][3]; out:
// [3][nh][nw]. Same size: a copy.
std::vector<uint8_t> resize_bicubic_aa(const uint8_t* rgb, int h, int w, int nh, int nw, int threads = 8);

struct PreparedImage {
  int grid_h = 0, grid_w = 0;      // the patch grid (both even)
  int tokens() const { return grid_h * grid_w / 4; }
  std::vector<uint16_t> patches;   // BF16 bits [grid_h * grid_w][3 * 2 * 16 * 16], merge-window order
  uint64_t hash = 0;               // of the decoded pixels and their size
};

// The whole path for one decoded image: resize, normalise ((x - 127.5) / 127.5 in FP32), patchify, BF16.
PreparedImage prepare_image(const DecodedImage& img, const ProcessorConfig& cfg = {});
uint64_t hash_pixels(const DecodedImage& img);

// Each distinct image's id in the prompt, 1,000,000 + k: one id per hash for the process's lifetime, never two
// images on one id (§20.7). Thread-safe.
class ImageRegistry {
 public:
  static constexpr int kFirstId = 1000000;
  int id_for(uint64_t hash);
  static bool is_image_id(int token) { return token >= kFirstId; }

 private:
  std::mutex mu_;
  std::unordered_map<uint64_t, int> ids_;
  int next_ = kFirstId;
};

}  // namespace ling::serve
