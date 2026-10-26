// The CPU half of images (specs/DREAMFERENCE_LING_ENGINE_VISION.md §20.8): smart_resize against transformers'
// values, data: URLs and format sniffing, the decoders on images encoded here, the resampler's identity and an
// independent reference, the patch order and normalisation, the registry, the expansion of the placeholders, the
// template's placeholders and URL order on both routes, and the M-RoPE table against get_rope_index's rule. No GPU and
// no model; the comparisons with production's own pixels need its image (ling-run --vision-check).
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <jpeglib.h>
#include <png.h>
#include <webp/encode.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/engine.hpp"
#include "serve/api.hpp"
#include "serve/image_input.hpp"

using json = nlohmann::ordered_json;
using namespace ling::serve;

namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

std::string b64(const std::string& bytes) {
  static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  size_t i = 0;
  for (; i + 2 < bytes.size(); i += 3) {
    const uint32_t v = (uint8_t(bytes[i]) << 16) | (uint8_t(bytes[i + 1]) << 8) | uint8_t(bytes[i + 2]);
    for (int k = 18; k >= 0; k -= 6) out += t[(v >> k) & 63];
  }
  if (i < bytes.size()) {
    uint32_t v = uint8_t(bytes[i]) << 16;
    if (i + 1 < bytes.size()) v |= uint8_t(bytes[i + 1]) << 8;
    out += t[(v >> 18) & 63];
    out += t[(v >> 12) & 63];
    out += i + 1 < bytes.size() ? t[(v >> 6) & 63] : '=';
    out += '=';
  }
  return out;
}

void png_write_mem(png_structp png, png_bytep data, png_size_t len) {
  static_cast<std::string*>(png_get_io_ptr(png))->append(reinterpret_cast<const char*>(data), len);
}
void png_flush_mem(png_structp) {}
std::string encode_png(const std::vector<uint8_t>& px, int w, int h, int color_type, int channels) {
  std::string out;
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  png_infop info = png_create_info_struct(png);
  png_set_write_fn(png, &out, png_write_mem, png_flush_mem);
  png_set_IHDR(png, info, w, h, 8, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  for (int y = 0; y < h; ++y) png_write_row(png, const_cast<png_bytep>(px.data() + size_t(y) * w * channels));
  png_write_end(png, nullptr);
  png_destroy_write_struct(&png, &info);
  return out;
}

std::string encode_jpeg(const std::vector<uint8_t>& rgb, int w, int h) {
  jpeg_compress_struct c;
  jpeg_error_mgr err;
  c.err = jpeg_std_error(&err);
  jpeg_create_compress(&c);
  unsigned char* buf = nullptr;
  unsigned long size = 0;
  jpeg_mem_dest(&c, &buf, &size);
  c.image_width = w;
  c.image_height = h;
  c.input_components = 3;
  c.in_color_space = JCS_RGB;
  jpeg_set_defaults(&c);
  jpeg_set_quality(&c, 90, TRUE);
  jpeg_start_compress(&c, TRUE);
  while (c.next_scanline < c.image_height) {
    JSAMPROW row = const_cast<JSAMPROW>(rgb.data() + size_t(c.next_scanline) * w * 3);
    jpeg_write_scanlines(&c, &row, 1);
  }
  jpeg_finish_compress(&c);
  std::string out(reinterpret_cast<char*>(buf), size);
  jpeg_destroy_compress(&c);
  free(buf);
  return out;
}

void test_smart_resize() {
  // transformers 5.12.1's smart_resize(h, w, factor=32, min_pixels=65536, max_pixels=16777216), computed in Python.
  const int cases[][4] = {{1080, 1920, 1088, 1920}, {1088, 1920, 1088, 1920}, {768, 1024, 768, 1024}, {48, 48, 256, 256},
                          {80, 80, 256, 256},       {16, 16, 256, 256},       {1, 200, 32, 3648},     {300, 2, 3136, 32},
                          {5000, 5000, 4096, 4096}, {4096, 4096, 4096, 4096}, {4097, 4096, 4096, 4096},
                          {2048, 1250, 2048, 1248}, {1250, 2048, 1248, 2048}, {10000, 300, 9984, 288},
                          {256, 256, 256, 256},     {240, 240, 256, 256},     {272, 272, 256, 256},
                          {208, 2000, 192, 1984},   {1440, 2560, 1440, 2560}, {2160, 3840, 2176, 3840},
                          {6000, 4000, 4992, 3328}, {100, 20000, 96, 20000},  {33, 33, 256, 256},
                          {96, 112, 256, 288},      {1200, 1600, 1216, 1600}, {4032, 3024, 4032, 3008},
                          {255, 255, 256, 256},
                          // halves round to even: 34.5 -> 34, 35.5 -> 36
                          {1104, 1920, 1088, 1920}, {1136, 1920, 1152, 1920}};
  for (const auto& c : cases) {
    int h = 0, w = 0;
    smart_resize(c[0], c[1], 32, 65536, 16777216, h, w);
    expect(h == c[2] && w == c[3], "smart_resize(" + std::to_string(c[0]) + ", " + std::to_string(c[1]) + ") = " +
                                       std::to_string(h) + "x" + std::to_string(w));
  }
  bool threw = false;
  try {
    int h, w;
    smart_resize(17, 4000, 32, 65536, 16777216, h, w);
  } catch (const ImageError&) {
    threw = true;
  }
  expect(threw, "an aspect ratio above 200 is refused");
}

void test_urls_and_formats() {
  expect(image_bytes_from_url("data:image/png;base64," + b64("hello")) == "hello", "data: URL");
  expect(image_bytes_from_url(b64("abc")) == "abc", "bare base64");
  for (const char* u : {"https://x/y.png", "http://x", "file:///etc/passwd", "/tmp/a.png"}) {
    std::string code;
    try {
      image_bytes_from_url(u);
    } catch (const ImageError& e) {
      code = e.code;
    }
    expect(code == "image_url_not_supported", std::string("refused: ") + u);
  }
  expect(sniff_image_format("\x89PNG\r\n\x1a\nxxxx") == ImageFormat::Png, "sniff PNG");
  expect(sniff_image_format("\xff\xd8\xff\xe0") == ImageFormat::Jpeg, "sniff JPEG");
  expect(sniff_image_format(std::string("RIFF\0\0\0\0WEBPVP8 ", 16)) == ImageFormat::Webp, "sniff WebP");
  expect(sniff_image_format("GIF89a....") == ImageFormat::Gif, "sniff GIF");
  std::string code;
  try {
    decode_image("GIF89a....");
  } catch (const ImageError& e) {
    code = e.code;
  }
  expect(code == "unsupported_image_format", "GIF refused");
}

void test_decoders() {
  const int w = 7, h = 5;
  std::vector<uint8_t> rgba(size_t(w) * h * 4), rgb(size_t(w) * h * 3);
  for (int i = 0; i < w * h; ++i)
    for (int c = 0; c < 4; ++c) {
      rgba[i * 4 + c] = uint8_t((i * 37 + c * 91) & 0xff);
      if (c < 3) rgb[i * 3 + c] = rgba[i * 4 + c];
    }
  // PNG RGBA: alpha dropped, not composited.
  DecodedImage a = decode_image(encode_png(rgba, w, h, PNG_COLOR_TYPE_RGB_ALPHA, 4));
  expect(a.width == w && a.height == h && a.rgb == rgb, "PNG RGBA decodes to its RGB, alpha dropped");
  // PNG grey: replicated.
  std::vector<uint8_t> grey(size_t(w) * h);
  for (int i = 0; i < w * h; ++i) grey[i] = uint8_t(i * 7);
  DecodedImage g = decode_image(encode_png(grey, w, h, PNG_COLOR_TYPE_GRAY, 1));
  bool ok = g.rgb.size() == size_t(w) * h * 3;
  for (int i = 0; ok && i < w * h; ++i) ok = g.rgb[i * 3] == grey[i] && g.rgb[i * 3 + 1] == grey[i] && g.rgb[i * 3 + 2] == grey[i];
  expect(ok, "PNG grey decodes to grey RGB");
  // WebP lossless with alpha.
  uint8_t* out = nullptr;
  const size_t n = WebPEncodeLosslessRGBA(rgba.data(), w, h, w * 4, &out);
  DecodedImage wp = decode_image(std::string(reinterpret_cast<char*>(out), n));
  WebPFree(out);
  expect(wp.width == w && wp.height == h && wp.rgb == rgb, "WebP lossless decodes exactly, alpha dropped");
  // JPEG (CPU decoder; the GPU one needs a device): the size and a plausible picture.
  std::vector<uint8_t> big(size_t(64) * 48 * 3);
  for (size_t i = 0; i < big.size(); ++i) big[i] = uint8_t((i / 3) % 64 * 4);
  DecodedImage j = decode_jpeg_cpu(encode_jpeg(big, 64, 48));
  int worst = 0;
  for (size_t i = 0; i < big.size(); ++i) worst = std::max(worst, std::abs(int(j.rgb[i]) - int(big[i])));
  expect(j.width == 64 && j.height == 48 && worst < 40, "JPEG decodes (worst " + std::to_string(worst) + ")");
}

// An independent antialiased bicubic in double (PIL's algorithm, which PyTorch's _aa kernels follow).
std::vector<double> ref_resize_1d(const std::vector<double>& in, int out) {
  const int n = static_cast<int>(in.size());
  const double scale = double(n) / out, support = scale >= 1 ? 2 * scale : 2;
  std::vector<double> o(out);
  for (int i = 0; i < out; ++i) {
    const double center = scale * (i + 0.5);
    const int xmin = std::max(int(center - support + 0.5), 0), xmax = std::min(int(center + support + 0.5), n);
    double tot = 0, acc = 0;
    for (int x = xmin; x < xmax; ++x) {
      double t = std::abs((x - center + 0.5) / (scale >= 1 ? scale : 1));
      const double a = -0.5;
      const double wgt = t < 1 ? ((a + 2) * t - (a + 3)) * t * t + 1 : t < 2 ? (((t - 5) * t + 8) * t - 4) * a : 0;
      tot += wgt;
      acc += wgt * in[x];
    }
    o[i] = acc / tot;
  }
  return o;
}

void test_resize() {
  const int h = 9, w = 13;
  std::vector<uint8_t> rgb(size_t(h) * w * 3);
  for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = uint8_t((i * 53 + 17) & 0xff);
  std::vector<uint8_t> same = resize_bicubic_aa(rgb.data(), h, w, h, w);
  bool ok = true;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int c = 0; c < 3; ++c) ok = ok && same[(size_t(c) * h + y) * w + x] == rgb[(size_t(y) * w + x) * 3 + c];
  expect(ok, "resize to the same size is a copy, HWC to CHW");
  for (auto [nh, nw] : {std::pair{32, 32}, std::pair{4, 6}, std::pair{9, 40}, std::pair{20, 13}}) {
    std::vector<uint8_t> got = resize_bicubic_aa(rgb.data(), h, w, nh, nw);
    int worst = 0;
    for (int c = 0; c < 3; ++c) {
      std::vector<std::vector<double>> rows(h);
      for (int y = 0; y < h; ++y) {
        std::vector<double> line(w);
        for (int x = 0; x < w; ++x) line[x] = rgb[(size_t(y) * w + x) * 3 + c];
        rows[y] = ref_resize_1d(line, nw);
      }
      for (int x = 0; x < nw; ++x) {
        std::vector<double> col(h);
        for (int y = 0; y < h; ++y) col[y] = rows[y][x];
        const std::vector<double> o = ref_resize_1d(col, nh);
        for (int y = 0; y < nh; ++y) {
          const int ref = int(std::nearbyint(std::min(255.0, std::max(0.0, o[y]))));
          worst = std::max(worst, std::abs(ref - int(got[(size_t(c) * nh + y) * nw + x])));
        }
      }
    }
    expect(worst <= 1, "resize to " + std::to_string(nh) + "x" + std::to_string(nw) + " within 1 level of the reference (" +
                           std::to_string(worst) + ")");
  }
}

float bf16f(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

void test_patches() {
  // A 256 x 320 image (already a multiple of 32, above the minimum): no resampling; a pixel's value encodes its place.
  DecodedImage img;
  img.width = 320;
  img.height = 256;
  img.rgb.resize(size_t(320) * 256 * 3);
  for (int y = 0; y < 256; ++y)
    for (int x = 0; x < 320; ++x)
      for (int c = 0; c < 3; ++c) img.rgb[(size_t(y) * 320 + x) * 3 + c] = uint8_t((y * 3 + x * 5 + c * 11) & 0xff);
  PreparedImage p = prepare_image(img);
  expect(p.grid_h == 16 && p.grid_w == 20 && p.tokens() == 80, "grid of a 256 x 320 image");
  auto norm = [](int v) { return (float(v) - 127.5f) / 127.5f; };
  // Patch k: window (k / 4) in the 8 x 10 merged grid, then (mh, mw) inside it.
  bool ok = true;
  for (int k : {0, 1, 2, 3, 4, 77, 319}) {
    const int win = k / 4, in = k % 4, bh = win / 10, bw = win % 10;
    const int y0 = (bh * 2 + in / 2) * 16, x0 = (bw * 2 + in % 2) * 16;
    for (int c = 0; c < 3; ++c)
      for (int t = 0; t < 2; ++t)
        for (int py : {0, 7, 15})
          for (int px : {0, 9, 15}) {
            const float want = norm(img.rgb[(size_t(y0 + py) * 320 + x0 + px) * 3 + c]);
            const float got = bf16f(p.patches[size_t(k) * 1536 + ((c * 2 + t) * 16 + py) * 16 + px]);
            ok = ok && std::abs(got - want) <= std::abs(want) * 0.004f + 1e-6f;
          }
  }
  expect(ok, "patches in merge-window order, (c, t, row, col) inside, normalised");
  DecodedImage img2 = img;
  expect(hash_pixels(img) == hash_pixels(img2), "the same pixels hash the same");
  img2.rgb[12345] ^= 1;
  expect(hash_pixels(img) != hash_pixels(img2), "one changed value changes the hash");
  ImageRegistry reg;
  const int a = reg.id_for(hash_pixels(img)), b = reg.id_for(hash_pixels(img2)), c = reg.id_for(hash_pixels(img));
  expect(a == ling::kImageTokenBase && b == a + 1 && c == a, "registry: one id per image, from 1,000,000");
}

void test_template_and_expansion() {
  const std::string png = "data:image/png;base64," + b64(encode_png(std::vector<uint8_t>(64 * 64 * 3, 128), 64, 64,
                                                                    PNG_COLOR_TYPE_RGB, 3));
  // Chat completions: images in user and tool messages, in order; vision off renders [image].
  json body = json::parse(R"({"messages":[{"role":"user","content":[{"type":"text","text":"A"},
      {"type":"image_url","image_url":{"url":"U1"}},{"type":"text","text":"B"}]},
      {"role":"assistant","content":"","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}]},
      {"role":"tool","tool_call_id":"c","content":[{"type":"text","text":"T"},{"type":"image_url","image_url":"U2"}]}]})");
  Request r = parse_request(body, true, true);
  expect(r.image_urls == std::vector<std::string>{"U1", "U2"}, "chat: image URLs in order");
  expect(r.prompt_text.find("A<|vision_start|><|image_pad|><|vision_end|>B") != std::string::npos, "chat: user placeholder");
  expect(r.prompt_text.find("<tool_response>\nT<|vision_start|><|image_pad|><|vision_end|>\n</tool_response>") !=
             std::string::npos,
         "chat: tool placeholder");
  Request off = parse_request(body, true, false);
  expect(off.image_urls.empty() && off.prompt_text.find("A[image]B") != std::string::npos, "vision off: [image]");
  json sys = json::parse(R"({"messages":[{"role":"system","content":[{"type":"image_url","image_url":{"url":"U"}}]},
      {"role":"user","content":"hi"}]})");
  bool threw = false;
  try {
    parse_request(sys, true, true);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  expect(threw, "a system message with an image is refused");
  // Responses: input_image in a message and in a function_call_output list (rendered, VISION §20.6).
  json resp = json::parse(R"({"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"look"},
      {"type":"input_image","image_url":"U1","detail":"high"}]},
      {"type":"function_call","call_id":"c1","name":"view_image","arguments":"{\"path\":\"a.png\"}"},
      {"type":"function_call_output","call_id":"c1","output":[{"type":"input_image","image_url":"U2"}]},
      {"type":"function_call","call_id":"c2","name":"sh","arguments":"{}"},
      {"type":"function_call_output","call_id":"c2","output":[{"type":"input_text","text":"x"},{"type":"input_text","text":"y"}]}]})");
  Request rr = parse_responses_request(resp, true);
  expect(rr.image_urls == std::vector<std::string>{"U1", "U2"}, "responses: input_image and the tool result's image");
  expect(rr.prompt_text.find("<tool_response>\n<|vision_start|><|image_pad|><|vision_end|>\n</tool_response>") !=
             std::string::npos,
         "responses: the tool result's image renders");
  expect(rr.prompt_text.find("<tool_response>\nxy\n</tool_response>") != std::string::npos,
         "responses: a text-only output list stays its text");
  Request rr_off = parse_responses_request(resp, false);
  expect(rr_off.image_urls.empty(), "responses without vision: no images");
  // Expansion: two placeholders, the same image twice, one id.
  const int PAD = 248056;
  std::vector<int> prompt = {1, 2, PAD, 3, PAD, 4};
  std::vector<ling::PromptImage> images;
  ImageRegistry reg;
  attach_images(prompt, PAD, {png, png}, reg, ProcessorConfig{}, images);
  // 64 x 64 is below the minimum: resized to 256 x 256, a 16 x 16 patch grid, 64 tokens.
  expect(images.size() == 1 && images[0].grid_h == 16 && images[0].tokens() == 64, "one image, 64 tokens");
  expect(prompt.size() == 4 + 128 && prompt[2] == ling::kImageTokenBase && prompt[66] == 3 &&
             prompt[67] == ling::kImageTokenBase,
         "each placeholder expanded into the image's tokens");
  std::string code;
  try {
    std::vector<int> p2 = {PAD};
    attach_images(p2, PAD, {}, reg, ProcessorConfig{}, images);
  } catch (const ImageError& e) {
    code = e.code;
  }
  expect(code == "invalid_image", "placeholders and images must match");
}

void test_rope_rows() {
  // [a, b, <vs>, image 2 x 3 merged (6), <ve>, c, <vs>, image 4 x 1 (4), <ve>, d]: transformers' get_rope_index gives
  // text at its running position, an image's (t, h, w) = (P, P + r, P + c), then P + max(rows, cols).
  ling::Engine::PrefillCursor cur;
  ling::PromptImage i1, i2;
  i1.token = ling::kImageTokenBase;
  i1.grid_h = 4;
  i1.grid_w = 6;
  i2.token = ling::kImageTokenBase + 1;
  i2.grid_h = 8;
  i2.grid_w = 2;
  cur.images = {i1, i2};
  cur.spans.push_back({3, 6, 0, 2, 3, 3});
  // a0 b1 vs2 | image base 3 | ve 6 c 7 vs 8 (3 + max(2, 3) = 6) | image base 9 | ve 13 d 14 (9 + max(4, 1) = 13)
  cur.spans.push_back({12, 4, 1, 4, 1, 9});
  std::vector<int> t;
  ling::Engine::rope_rows(cur, 0, 18, t);
  const int want[18][3] = {{0, 0, 0}, {1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {3, 3, 4}, {3, 3, 5}, {3, 4, 3}, {3, 4, 4}, {3, 4, 5},
                           {6, 6, 6}, {7, 7, 7}, {8, 8, 8}, {9, 9, 9}, {9, 10, 9}, {9, 11, 9}, {9, 12, 9}, {13, 13, 13},
                           {14, 14, 14}};
  bool ok = true;
  for (int i = 0; i < 18; ++i)
    for (int k = 0; k < 3; ++k) ok = ok && t[size_t(i) * 3 + k] == want[i][k];
  expect(ok, "M-RoPE table across two images");
  std::vector<int> part;
  ling::Engine::rope_rows(cur, 5, 14, part);
  ok = true;
  for (int i = 5; i < 14; ++i)
    for (int k = 0; k < 3; ++k) ok = ok && part[size_t(i - 5) * 3 + k] == want[i][k];
  expect(ok, "a chunk that starts inside an image");
  // After the prompt (18 tokens, last position 14): delta = 15 - 18 = -3.
  expect(ling::Engine::rope_delta_after(cur) == -3, "the delta after the prompt");
}

}  // namespace

int main() {
  set_jpeg_decoder(JpegDecoder::Cpu);
  test_smart_resize();
  test_urls_and_formats();
  test_decoders();
  test_resize();
  test_patches();
  test_template_and_expansion();
  test_rope_rows();
  std::printf(failures ? "VISION INPUT TESTS FAILED (%d)\n" : "all vision input tests passed\n", failures);
  return failures ? 1 : 0;
}
