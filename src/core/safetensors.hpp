// Reads a sharded safetensors checkpoint (model.safetensors.index.json plus its shards) by mapping each
// shard into memory; tensors are views into the mapping until they are copied to the GPU.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ling {

struct TensorView {
  std::string dtype;            // "BF16", "F32", "F8_E4M3", "U8", ...
  std::vector<int64_t> shape;
  const uint8_t* data = nullptr;
  size_t bytes = 0;

  int64_t numel() const;
};

class MappedFile {
 public:
  explicit MappedFile(const std::string& path);
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  int fd_ = -1;
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

class Checkpoint {
 public:
  // `dir` holds config.json, model.safetensors.index.json and the shards.
  explicit Checkpoint(const std::string& dir);

  bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  const TensorView& get(const std::string& name) const;
  const std::string& dir() const { return dir_; }
  size_t count() const { return tensors_.size(); }

 private:
  std::string dir_;
  std::vector<std::unique_ptr<MappedFile>> files_;
  std::map<std::string, TensorView> tensors_;
};

}  // namespace ling
