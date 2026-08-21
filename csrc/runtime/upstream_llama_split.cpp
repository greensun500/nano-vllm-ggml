// Minimal fixed upstream dependency for llama_model_loader.
//
// Source: third_party/llama.cpp/src/llama.cpp (model split helpers).  Keep
// this implementation alongside the fixed tokenizer subset instead of linking
// the complete llama.cpp runtime merely to parse an optional split GGUF name.

#include <llama.h>

#include <algorithm>
#include <cstdio>
#include <string>

int32_t llama_split_path(
    char * split_path,
    size_t maxlen,
    const char * path_prefix,
    int32_t split_no,
    int32_t split_count) {
    static const char * const split_path_format = "%s-%05d-of-%05d.gguf";
    const int written = std::snprintf(
        split_path, maxlen, split_path_format, path_prefix, split_no + 1, split_count);
    if (written < 0 || static_cast<size_t>(written) >= maxlen) {
        return 0;
    }
    return static_cast<int32_t>(written);
}

int32_t llama_split_prefix(
    char * split_prefix,
    size_t maxlen,
    const char * split_path,
    int32_t split_no,
    int32_t split_count) {
    const std::string path(split_path);
    char postfix[32];
    std::snprintf(postfix, sizeof(postfix), "-%05d-of-%05d.gguf", split_no + 1, split_count);
    const std::string suffix(postfix);
    if (path.size() <= suffix.size()) {
        return 0;
    }
    const size_t prefix_size = path.size() - suffix.size();
    if (path.compare(prefix_size, std::string::npos, suffix) != 0) {
        return 0;
    }
    const size_t copy_len = std::min(prefix_size + 1, maxlen);
    std::snprintf(split_prefix, copy_len, "%s", split_path);
    return static_cast<int32_t>(prefix_size);
}
