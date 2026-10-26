// Images from a request to the tower's input (image_input.hpp).
#include "serve/image_input.hpp"

#include <cuda_runtime.h>
#include <jpeglib.h>
#include <nvjpeg.h>
#include <png.h>
#include <webp/decode.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <csetjmp>
#include <cstring>
#include <thread>

namespace ling::serve {
namespace {

std::atomic<JpegDecoder> g_jpeg{JpegDecoder::Gpu};

int b64v(unsigned char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+' || c == '-') return 62;
  if (c == '/' || c == '_') return 63;
  return -1;
}

std::string b64decode(const std::string& s, size_t from, size_t max_bytes) {
  std::string out;
  out.reserve((s.size() - from) * 3 / 4);
  uint32_t acc = 0;
  int bits = 0;
  for (size_t i = from; i < s.size(); ++i) {
    const unsigned char c = s[i];
    if (c == '=') break;
    if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
    const int v = b64v(c);
    if (v < 0) throw ImageError("the image's base64 payload is malformed", "invalid_image");
    acc = (acc << 6) | uint32_t(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<char>((acc >> bits) & 0xff));
      if (out.size() > max_bytes)
        throw ImageError("the image is larger than " + std::to_string(max_bytes >> 20) + " MB", "image_too_large");
    }
  }
  return out;
}

bool starts(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

// ---- PNG (libpng, low-level: palette and low bit depths expanded, 16 bits to their high byte, grey to RGB, alpha
// stripped without compositing, as Pillow's convert("RGB") leaves them; no gamma). ----
struct PngSrc {
  const uint8_t* p;
  size_t n, off;
};
void png_read_mem(png_structp png, png_bytep out, png_size_t len) {
  auto* s = static_cast<PngSrc*>(png_get_io_ptr(png));
  if (s->off + len > s->n) png_error(png, "truncated");
  std::memcpy(out, s->p + s->off, len);
  s->off += len;
}
DecodedImage decode_png(const std::string& bytes) {
  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  png_infop info = png ? png_create_info_struct(png) : nullptr;
  if (!png || !info) throw std::runtime_error("libpng: out of memory");
  DecodedImage img;
  std::vector<png_bytep> rows;
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_read_struct(&png, &info, nullptr);
    throw ImageError("the PNG image could not be decoded", "invalid_image");
  }
  PngSrc src{reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), 0};
  png_set_read_fn(png, &src, png_read_mem);
  png_set_user_limits(png, 1u << 16, 1u << 16);
  png_read_info(png, info);
  png_set_expand(png);
  png_set_strip_16(png);
  png_set_gray_to_rgb(png);
  png_set_strip_alpha(png);
  png_set_interlace_handling(png);
  png_read_update_info(png, info);
  img.width = static_cast<int>(png_get_image_width(png, info));
  img.height = static_cast<int>(png_get_image_height(png, info));
  if (png_get_rowbytes(png, info) != size_t(img.width) * 3) png_error(png, "unexpected row size");
  img.rgb.resize(size_t(img.width) * img.height * 3);
  rows.resize(img.height);
  for (int y = 0; y < img.height; ++y) rows[y] = img.rgb.data() + size_t(y) * img.width * 3;
  png_read_image(png, rows.data());
  png_read_end(png, nullptr);
  png_destroy_read_struct(&png, &info, nullptr);
  return img;
}

// ---- WebP (libwebp: its default decoder, fancy upsampling; alpha dropped). ----
DecodedImage decode_webp(const std::string& bytes) {
  WebPBitstreamFeatures f;
  const auto* p = reinterpret_cast<const uint8_t*>(bytes.data());
  if (WebPGetFeatures(p, bytes.size(), &f) != VP8_STATUS_OK)
    throw ImageError("the WebP image could not be decoded", "invalid_image");
  if (f.has_animation) throw ImageError("animated WebP images are not supported", "unsupported_image_format");
  DecodedImage img;
  img.width = f.width;
  img.height = f.height;
  img.rgb.resize(size_t(f.width) * f.height * 3);
  if (!WebPDecodeRGBInto(p, bytes.size(), img.rgb.data(), img.rgb.size(), f.width * 3))
    throw ImageError("the WebP image could not be decoded", "invalid_image");
  return img;
}

// ---- JPEG on the CPU (libjpeg-turbo with Pillow's settings: ISLOW DCT, fancy upsampling, RGB out). ----
struct JpegErr {
  jpeg_error_mgr mgr;
  jmp_buf jb;
};
void jpeg_fail(j_common_ptr c) { longjmp(reinterpret_cast<JpegErr*>(c->err)->jb, 1); }

// ---- JPEG on the GPU (nvJPEG, as torchvision.io.decode_jpeg(device="cuda") in production). ----
struct Nvjpeg {
  nvjpegHandle_t h = nullptr;
  nvjpegJpegState_t st = nullptr;
  cudaStream_t stream = nullptr;
  bool ok = false;
  Nvjpeg() {
    if (nvjpegCreateSimple(&h) != NVJPEG_STATUS_SUCCESS) return;
    if (nvjpegJpegStateCreate(h, &st) != NVJPEG_STATUS_SUCCESS) return;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return;
    ok = true;
  }
};
std::mutex g_nvjpeg_mu;

bool decode_jpeg_gpu(const std::string& bytes, DecodedImage& img) {
  std::lock_guard<std::mutex> lock(g_nvjpeg_mu);
  static Nvjpeg nv;
  if (!nv.ok) return false;
  const auto* p = reinterpret_cast<const unsigned char*>(bytes.data());
  int nc = 0;
  nvjpegChromaSubsampling_t sub;
  int w[NVJPEG_MAX_COMPONENT], h[NVJPEG_MAX_COMPONENT];
  if (nvjpegGetImageInfo(nv.h, p, bytes.size(), &nc, &sub, w, h) != NVJPEG_STATUS_SUCCESS) return false;
  if (w[0] <= 0 || h[0] <= 0 || size_t(w[0]) * h[0] > (size_t(1) << 28)) return false;
  img.width = w[0];
  img.height = h[0];
  const size_t plane = size_t(w[0]) * h[0];
  uint8_t* d = nullptr;
  if (cudaMalloc(&d, plane * 3) != cudaSuccess) return false;
  nvjpegImage_t out{};
  for (int c = 0; c < 3; ++c) {
    out.channel[c] = d + plane * c;
    out.pitch[c] = static_cast<unsigned>(w[0]);
  }
  bool good = nvjpegDecode(nv.h, nv.st, p, bytes.size(), NVJPEG_OUTPUT_RGB, &out, nv.stream) == NVJPEG_STATUS_SUCCESS;
  std::vector<uint8_t> planar;
  if (good) {
    planar.resize(plane * 3);
    good = cudaMemcpyAsync(planar.data(), d, plane * 3, cudaMemcpyDeviceToHost, nv.stream) == cudaSuccess &&
           cudaStreamSynchronize(nv.stream) == cudaSuccess;
  }
  cudaFree(d);
  if (!good) return false;
  img.rgb.resize(plane * 3);
  for (size_t i = 0; i < plane; ++i)
    for (int c = 0; c < 3; ++c) img.rgb[i * 3 + c] = planar[plane * c + i];
  return true;
}

uint16_t bf16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x7fffffu)) return static_cast<uint16_t>((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even
  return static_cast<uint16_t>(u >> 16);
}

// PyTorch's BicubicFilterFunctor (a = -0.5), with the multiply-adds nvcc contracts.
float bicubic(float x) {
  const float a = -0.5f;
  if (x < 0) x = -x;
  if (x < 1) return std::fmaf(((a + 2) * x - (a + 3)) * x, x, 1.f);
  if (x < 2) return std::fmaf(std::fmaf(x - 5, x, 8.f), x, -4.f) * a;
  return 0.f;
}

// One dimension's spans and normalised weights, as _compute_weights_span and _compute_weights compute them.
struct Taps {
  std::vector<int> min, size;
  std::vector<float> w;  // [out][interp]
  int interp = 0;
};
Taps taps(int in, int out) {
  Taps t;
  const float scale = static_cast<float>(in) / static_cast<float>(out);
  const float support = scale >= 1.f ? (4 * 0.5f) * scale : 4 * 0.5f;
  t.interp = static_cast<int>(std::ceil(support)) * 2 + 1;
  t.min.resize(out);
  t.size.resize(out);
  t.w.assign(size_t(out) * t.interp, 0.f);
  const float invscale = scale >= 1.f ? static_cast<float>(1.0 / scale) : 1.f;
  for (int i = 0; i < out; ++i) {
    const float center = static_cast<float>(scale * (i + 0.5));
    const int xmin = std::max(static_cast<int>(center - support + 0.5), 0);
    const int xsize = std::min(static_cast<int>(center + support + 0.5), in) - xmin;
    t.min[i] = xmin;
    t.size[i] = xsize;
    const float xmin_m_center = static_cast<float>(xmin) - center;
    float total = 0.f;
    float* w = t.w.data() + size_t(i) * t.interp;
    for (int j = 0; j < xsize; ++j) {
      const float v = bicubic((static_cast<float>(j) + xmin_m_center + 0.5f) * invscale);
      w[j] = v;
      total += v;
    }
    if (total != 0.f)
      for (int j = 0; j < xsize; ++j) w[j] /= total;
  }
  return t;
}

template <typename F>
void parallel_rows(int rows, int threads, F f) {
  threads = std::max(1, std::min(threads, rows));
  std::vector<std::thread> pool;
  for (int k = 0; k < threads; ++k)
    pool.emplace_back([&, k] {
      for (int r = k; r < rows; r += threads) f(r);
    });
  for (auto& th : pool) th.join();
}

}  // namespace

std::string image_bytes_from_url(const std::string& url, size_t max_bytes) {
  if (starts(url, "http://") || starts(url, "https://") || starts(url, "file:") || starts(url, "/"))
    throw ImageError("only data: URLs are accepted for images (this server does not fetch URLs or read files)",
                     "image_url_not_supported");
  if (starts(url, "data:")) {
    const size_t comma = url.find(',');
    if (comma == std::string::npos) throw ImageError("the image's data: URL has no payload", "invalid_image");
    const std::string meta = url.substr(5, comma - 5);
    if (meta.find(";base64") == std::string::npos)
      throw ImageError("the image's data: URL is not base64", "invalid_image");
    return b64decode(url, comma + 1, max_bytes);
  }
  return b64decode(url, 0, max_bytes);
}

ImageFormat sniff_image_format(const std::string& b) {
  auto at = [&](size_t i, const char* s) { return b.size() >= i + std::strlen(s) && b.compare(i, std::strlen(s), s) == 0; };
  if (at(0, "\x89PNG\r\n\x1a\n")) return ImageFormat::Png;
  if (b.size() >= 3 && uint8_t(b[0]) == 0xff && uint8_t(b[1]) == 0xd8 && uint8_t(b[2]) == 0xff) return ImageFormat::Jpeg;
  if (at(0, "RIFF") && at(8, "WEBP")) return ImageFormat::Webp;
  if (at(0, "GIF87a") || at(0, "GIF89a")) return ImageFormat::Gif;
  if (at(0, "BM")) return ImageFormat::Bmp;
  if (at(0, "II*") || at(0, "MM\0*")) return ImageFormat::Tiff;
  if (at(4, "ftyp")) return ImageFormat::Heif;
  return ImageFormat::Unknown;
}

const char* image_format_name(ImageFormat f) {
  switch (f) {
    case ImageFormat::Png: return "PNG";
    case ImageFormat::Jpeg: return "JPEG";
    case ImageFormat::Webp: return "WebP";
    case ImageFormat::Gif: return "GIF";
    case ImageFormat::Bmp: return "BMP";
    case ImageFormat::Tiff: return "TIFF";
    case ImageFormat::Heif: return "HEIF/AVIF";
    default: return "unknown";
  }
}

void set_jpeg_decoder(JpegDecoder d) { g_jpeg = d; }

DecodedImage decode_jpeg_cpu(const std::string& bytes) {
  jpeg_decompress_struct cinfo;
  JpegErr err;
  cinfo.err = jpeg_std_error(&err.mgr);
  err.mgr.error_exit = jpeg_fail;
  DecodedImage img;
  if (setjmp(err.jb)) {
    jpeg_destroy_decompress(&cinfo);
    throw ImageError("the JPEG image could not be decoded", "invalid_image");
  }
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size());
  jpeg_read_header(&cinfo, TRUE);
  cinfo.out_color_space = JCS_RGB;
  cinfo.dct_method = JDCT_ISLOW;
  cinfo.do_fancy_upsampling = TRUE;
  jpeg_start_decompress(&cinfo);
  img.width = static_cast<int>(cinfo.output_width);
  img.height = static_cast<int>(cinfo.output_height);
  img.rgb.resize(size_t(img.width) * img.height * 3);
  while (cinfo.output_scanline < cinfo.output_height) {
    JSAMPROW row = img.rgb.data() + size_t(cinfo.output_scanline) * img.width * 3;
    jpeg_read_scanlines(&cinfo, &row, 1);
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  return img;
}

DecodedImage decode_image(const std::string& bytes) {
  const ImageFormat f = sniff_image_format(bytes);
  switch (f) {
    case ImageFormat::Png: return decode_png(bytes);
    case ImageFormat::Webp: return decode_webp(bytes);
    case ImageFormat::Jpeg: {
      DecodedImage img;
      // Production sends a JPEG to nvJPEG only when it also ends with EOI (is_jpeg_with_cuda).
      const bool eoi = bytes.size() >= 2 && uint8_t(bytes[bytes.size() - 2]) == 0xff && uint8_t(bytes.back()) == 0xd9;
      if (g_jpeg == JpegDecoder::Gpu && eoi && decode_jpeg_gpu(bytes, img)) return img;
      return decode_jpeg_cpu(bytes);
    }
    case ImageFormat::Unknown:
      throw ImageError("the image's format is not recognised (PNG, JPEG and WebP are accepted)", "unsupported_image_format");
    default:
      throw ImageError(std::string(image_format_name(f)) + " images are not accepted (PNG, JPEG and WebP are)",
                       "unsupported_image_format");
  }
}

void smart_resize(int height, int width, int factor, long min_pixels, long max_pixels, int& out_h, int& out_w) {
  if (height < 1 || width < 1) throw ImageError("the image is empty", "invalid_image");
  if (double(std::max(height, width)) / std::min(height, width) > 200)
    throw ImageError("the image's aspect ratio must be smaller than 200", "invalid_image");
  // Python's round(): halves to even (std::nearbyint in the default rounding mode).
  long h_bar = static_cast<long>(std::nearbyint(double(height) / factor)) * factor;
  long w_bar = static_cast<long>(std::nearbyint(double(width) / factor)) * factor;
  if (h_bar * w_bar > max_pixels) {
    const double beta = std::sqrt(double(height) * width / max_pixels);
    h_bar = std::max<long>(factor, static_cast<long>(std::floor(height / beta / factor)) * factor);
    w_bar = std::max<long>(factor, static_cast<long>(std::floor(width / beta / factor)) * factor);
  } else if (h_bar * w_bar < min_pixels) {
    const double beta = std::sqrt(double(min_pixels) / (double(height) * width));
    h_bar = static_cast<long>(std::ceil(height * beta / factor)) * factor;
    w_bar = static_cast<long>(std::ceil(width * beta / factor)) * factor;
  }
  out_h = static_cast<int>(h_bar);
  out_w = static_cast<int>(w_bar);
}

std::vector<uint8_t> resize_bicubic_aa(const uint8_t* rgb, int h, int w, int nh, int nw, int threads) {
  std::vector<uint8_t> out(size_t(3) * nh * nw);
  if (nh == h && nw == w) {
    for (size_t i = 0; i < size_t(h) * w; ++i)
      for (int c = 0; c < 3; ++c) out[size_t(c) * h * w + i] = rgb[i * 3 + c];
    return out;
  }
  const Taps tx = taps(w, nw), ty = taps(h, nh);
  // Horizontal pass: [3][h][nw] in FP32 (the kernel's buffer2 values, which depend only on the row and the output
  // column, so computing them once per row is the kernel's arithmetic).
  std::vector<float> mid(size_t(3) * h * nw);
  parallel_rows(h, threads, [&](int y) {
    for (int ox = 0; ox < nw; ++ox) {
      const int xmin = tx.min[ox], xs = tx.size[ox];
      const float* wx = tx.w.data() + size_t(ox) * tx.interp;
      for (int c = 0; c < 3; ++c) {
        const uint8_t* src = rgb + (size_t(y) * w + xmin) * 3 + c;
        float acc = float(src[0]) * wx[0];
        for (int j = 1; j < xs; ++j) acc = std::fmaf(float(src[size_t(j) * 3]), wx[j], acc);
        mid[(size_t(c) * h + y) * nw + ox] = acc;
      }
    }
  });
  parallel_rows(nh, threads, [&](int oy) {
    const int ymin = ty.min[oy], ys = ty.size[oy];
    const float* wy = ty.w.data() + size_t(oy) * ty.interp;
    for (int c = 0; c < 3; ++c)
      for (int ox = 0; ox < nw; ++ox) {
        const float* col = mid.data() + (size_t(c) * h + ymin) * nw + ox;
        float acc = col[0] * wy[0];
        for (int j = 1; j < ys; ++j) acc = std::fmaf(col[size_t(j) * nw], wy[j], acc);
        acc = std::min(std::max(acc, 0.f), 255.f);
        out[(size_t(c) * nh + oy) * nw + ox] = static_cast<uint8_t>(std::nearbyint(acc));
      }
  });
  return out;
}

uint64_t hash_pixels(const DecodedImage& img) {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](uint8_t b) {
    h ^= b;
    h *= 1099511628211ull;
  };
  for (int v : {img.width, img.height})
    for (int i = 0; i < 4; ++i) mix(static_cast<uint8_t>(v >> (8 * i)));
  // 8 bytes at a time with a second multiply round, then the tail.
  const uint8_t* p = img.rgb.data();
  const size_t n = img.rgb.size(), words = n / 8;
  for (size_t i = 0; i < words; ++i) {
    uint64_t v;
    std::memcpy(&v, p + 8 * i, 8);
    h = (h ^ v) * 0x9E3779B97F4A7C15ull;
    h ^= h >> 29;
  }
  for (size_t i = words * 8; i < n; ++i) mix(p[i]);
  return h;
}

PreparedImage prepare_image(const DecodedImage& img, const ProcessorConfig& cfg) {
  PreparedImage out;
  int nh = 0, nw = 0;
  smart_resize(img.height, img.width, cfg.factor(), cfg.min_pixels, cfg.max_pixels, nh, nw);
  const std::vector<uint8_t> chw = resize_bicubic_aa(img.rgb.data(), img.height, img.width, nh, nw);
  const int P = cfg.patch, M = cfg.merge, T = cfg.temporal;
  out.grid_h = nh / P;
  out.grid_w = nw / P;
  const int dim = 3 * T * P * P;
  out.patches.resize(size_t(out.grid_h) * out.grid_w * dim);
  // The normalised value of each uint8 level, once: (x - 127.5) / 127.5 in FP32, then BF16.
  uint16_t lut[256];
  for (int v = 0; v < 256; ++v) lut[v] = bf16_bits((static_cast<float>(v) - 127.5f) / 127.5f);
  const int wh = out.grid_h / M, ww = out.grid_w / M;
  parallel_rows(wh, 8, [&](int bh) {
    for (int bw = 0; bw < ww; ++bw)
      for (int mh = 0; mh < M; ++mh)
        for (int mw = 0; mw < M; ++mw) {
          const size_t p = ((size_t(bh) * ww + bw) * M + mh) * M + mw;
          uint16_t* dst = out.patches.data() + p * dim;
          const int y0 = (bh * M + mh) * P, x0 = (bw * M + mw) * P;
          for (int c = 0; c < 3; ++c)
            for (int t = 0; t < T; ++t)
              for (int py = 0; py < P; ++py)
                for (int px = 0; px < P; ++px)
                  dst[((c * T + t) * P + py) * P + px] = lut[chw[(size_t(c) * nh + y0 + py) * nw + x0 + px]];
        }
  });
  out.hash = hash_pixels(img);
  return out;
}

void attach_images(std::vector<int>& prompt, int image_pad, const std::vector<std::string>& urls, ImageRegistry& registry,
                   const ProcessorConfig& cfg, std::vector<PromptImage>& images) {
  int pads = 0;
  for (int t : prompt) pads += t == image_pad;
  if (pads != static_cast<int>(urls.size()))
    throw ImageError("the prompt holds " + std::to_string(pads) + " image placeholders for " + std::to_string(urls.size()) +
                         " images (is <|image_pad|> in the text?)",
                     "invalid_image");
  if (urls.size() > 64) throw ImageError("at most 64 images per request", "too_many_images");
  std::vector<int> ids;
  for (const std::string& url : urls) {
    const DecodedImage img = decode_image(image_bytes_from_url(url));
    PreparedImage prep = prepare_image(img, cfg);
    const int id = registry.id_for(prep.hash);
    ids.push_back(id);
    bool known = false;
    for (const PromptImage& p : images) known = known || p.token == id;
    if (!known) {
      PromptImage pi;
      pi.token = id;
      pi.grid_h = prep.grid_h;
      pi.grid_w = prep.grid_w;
      pi.patches = std::make_shared<const std::vector<uint16_t>>(std::move(prep.patches));
      images.push_back(std::move(pi));
    }
  }
  std::vector<int> out;
  out.reserve(prompt.size());
  size_t k = 0;
  for (int t : prompt) {
    if (t != image_pad) {
      out.push_back(t);
      continue;
    }
    const int id = ids[k++];
    for (const PromptImage& p : images)
      if (p.token == id) {
        out.insert(out.end(), p.tokens(), id);
        break;
      }
  }
  prompt.swap(out);
}

int ImageRegistry::id_for(uint64_t hash) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = ids_.find(hash);
  if (it != ids_.end()) return it->second;
  if (next_ == 0x7fffffff) throw std::runtime_error("image registry exhausted");
  const int id = next_++;
  ids_.emplace(hash, id);
  return id;
}

}  // namespace ling::serve
