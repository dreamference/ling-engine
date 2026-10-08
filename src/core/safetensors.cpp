#include "core/safetensors.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace ling {

int64_t TensorView::numel() const {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

MappedFile::MappedFile(const std::string& path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) throw std::runtime_error("cannot open " + path);
  struct stat st {};
  if (::fstat(fd_, &st) != 0) throw std::runtime_error("cannot stat " + path);
  size_ = static_cast<size_t>(st.st_size);
  void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (p == MAP_FAILED) throw std::runtime_error("cannot map " + path);
  data_ = static_cast<const uint8_t*>(p);
}

MappedFile::~MappedFile() {
  if (data_) ::munmap(const_cast<uint8_t*>(data_), size_);
  if (fd_ >= 0) ::close(fd_);
}

Checkpoint::Checkpoint(const std::string& dir) : dir_(dir) {
  std::ifstream index_file(dir + "/model.safetensors.index.json");
  if (!index_file) throw std::runtime_error("no model.safetensors.index.json in " + dir);
  nlohmann::json index = nlohmann::json::parse(index_file);
  std::set<std::string> shards;
  for (auto& [name, shard] : index["weight_map"].items()) shards.insert(shard.get<std::string>());

  for (const std::string& shard : shards) {
    auto file = std::make_unique<MappedFile>(dir + "/" + shard);
    if (file->size() < 8) throw std::runtime_error("truncated shard " + shard);
    uint64_t header_len = 0;
    std::memcpy(&header_len, file->data(), 8);
    if (8 + header_len > file->size()) throw std::runtime_error("bad header in " + shard);
    nlohmann::json header = nlohmann::json::parse(
        std::string(reinterpret_cast<const char*>(file->data() + 8), header_len));
    const uint8_t* base = file->data() + 8 + header_len;
    for (auto& [name, info] : header.items()) {
      if (name == "__metadata__") continue;
      TensorView view;
      view.dtype = info["dtype"].get<std::string>();
      view.shape = info["shape"].get<std::vector<int64_t>>();
      auto offsets = info["data_offsets"].get<std::vector<size_t>>();
      if (offsets.size() != 2 || offsets[1] < offsets[0] ||
          8 + header_len + offsets[1] > file->size()) {
        throw std::runtime_error("bad offsets for " + name);
      }
      view.data = base + offsets[0];
      view.bytes = offsets[1] - offsets[0];
      tensors_[name] = view;
    }
    files_.push_back(std::move(file));
  }
}

const TensorView& Checkpoint::get(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::runtime_error("tensor not in checkpoint: " + name);
  return it->second;
}

}  // namespace ling
