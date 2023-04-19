// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>
#include "paddle/include/paddle_inference_api.h"

int WARMUP_COUNT = 1;
int REPEAT_COUNT = 1;

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;
using paddle_infer::PrecisionType;

const std::vector<std::pair<std::string, std::string>> CANDIDATE_MODEL_FILES = {
    {"inference.pdmodel", "inference.pdiparams"},
    {"model.pdmodel", "model.pdiparams"},
    {"model", "params"}};

#ifdef __QNX__
#include <devctl.h>
#include <fcntl.h>
inline int64_t get_current_us() {
  auto fd = open("/dev/qgptp", O_RDONLY);
  if (fd < 0) {
    printf("open '/dev/qgptp' failed.");
  }
  uint64_t time_nsec;
#define GPTP_GETTIME __DIOF(_DCMD_MISC, 1, int)
  if (EOK != devctl(fd, GPTP_GETTIME, &time_nsec, sizeof(time_nsec), NULL)) {
    printf("devctl failed.");
  }
  if (close(fd) < 0) {
    printf("close fd failed.");
  }
  return time_nsec / 1000;
}
#else
inline int64_t get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}
#endif

template <typename T>
void get_value_from_sstream(std::stringstream *ss, T *value) {
  (*ss) >> (*value);
}

template <>
void get_value_from_sstream<std::string>(std::stringstream *ss,
                                         std::string *value) {
  *value = ss->str();
}

template <typename T>
std::vector<T> split_string(const std::string &str, char sep) {
  std::stringstream ss;
  std::vector<T> values;
  T value;
  values.clear();
  for (auto c : str) {
    if (c != sep) {
      ss << c;
    } else {
      get_value_from_sstream<T>(&ss, &value);
      values.push_back(std::move(value));
      ss.str({});
      ss.clear();
    }
  }
  if (!ss.str().empty()) {
    get_value_from_sstream<T>(&ss, &value);
    values.push_back(std::move(value));
    ss.str({});
    ss.clear();
  }
  return values;
}

bool read_file(const std::string &filename,
               std::vector<char> *contents,
               bool binary = true) {
  FILE *fp = fopen(filename.c_str(), binary ? "rb" : "r");
  if (!fp) return false;
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  contents->clear();
  contents->resize(size);
  size_t offset = 0;
  char *ptr = reinterpret_cast<char *>(&(contents->at(0)));
  while (offset < size) {
    size_t already_read = fread(ptr, 1, size - offset, fp);
    offset += already_read;
    ptr += already_read;
  }
  fclose(fp);
  return true;
}

bool write_file(const std::string &filename,
                const std::vector<char> &contents,
                bool binary = true) {
  FILE *fp = fopen(filename.c_str(), binary ? "wb" : "w");
  if (!fp) return false;
  size_t size = contents.size();
  size_t offset = 0;
  const char *ptr = reinterpret_cast<const char *>(&(contents.at(0)));
  while (offset < size) {
    size_t already_written = fwrite(ptr, 1, size - offset, fp);
    offset += already_written;
    ptr += already_written;
  }
  fclose(fp);
  return true;
}

// The helper functions for loading and running model from command line and
// verifying output data
std::vector<std::string> parse_types(std::string text) {
  std::vector<std::string> types;
  while (!text.empty()) {
    size_t index = text.find_first_of(":");
    std::string type = text.substr(0, index);
    std::cout << type << std::endl;
    types.push_back(type);
    if (index == std::string::npos) {
      break;
    } else {
      text = text.substr(index + 1);
    }
  }
  return types;
}

std::vector<std::vector<int32_t>> parse_shapes(std::string text) {
  std::vector<std::vector<int32_t>> shapes;
  while (!text.empty()) {
    size_t index = text.find_first_of(":");
    std::string slice = text.substr(0, index);
    std::vector<int32_t> shape;
    while (!slice.empty()) {
      size_t index = slice.find_first_of(",");
      int d = atoi(slice.substr(0, index).c_str());
      std::cout << d << std::endl;
      shape.push_back(d);
      if (index == std::string::npos) {
        break;
      } else {
        slice = slice.substr(index + 1);
      }
    }
    shapes.push_back(shape);
    if (index == std::string::npos) {
      break;
    } else {
      text = text.substr(index + 1);
    }
  }
  return shapes;
}

int64_t shape_production(std::vector<int32_t> shape) {
  int64_t s = 1;
  for (int32_t dim : shape) {
    s *= dim;
  }
  return s;
}

void fill_input_tensors(const std::shared_ptr<Predictor> predictor,
                        const std::vector<std::vector<int32_t>> &input_shapes,
                        const std::vector<std::string> &input_types,
                        const float value) {
#define FILL_TENSOR_WITH_TYPE(type)           \
  std::vector<type> input_data(input_size);   \
  for (int j = 0; j < input_size; j++) {      \
    input_data[j] = static_cast<type>(value); \
  }                                           \
  input_tensor->CopyFromCpu(input_data.data());
  auto input_names = predictor->GetInputNames();
  for (int i = 0; i < input_shapes.size(); i++) {
    auto input_tensor = predictor->GetInputHandle(input_names[i]);
    input_tensor->Reshape(input_shapes[i]);
    auto input_size = shape_production(input_tensor->shape());
    if (input_types[i] == "float32") {
      FILL_TENSOR_WITH_TYPE(float)
    } else if (input_types[i] == "int32") {
      FILL_TENSOR_WITH_TYPE(int32_t)
    } else if (input_types[i] == "int64") {
      FILL_TENSOR_WITH_TYPE(int64_t)
    } else {
      printf(
          "Unsupported input data type '%s', only 'float32', 'int32', 'int64' "
          "are supported!\n",
          input_types[i].c_str());
      exit(-1);
    }
  }
#undef FILL_TENSOR_WITH_TYPE
}

const int MAX_DISPLAY_OUTPUT_TENSOR_SIZE = 10000;
void print_output_tensors(const std::shared_ptr<Predictor> &predictor,
                          const std::vector<std::string> &output_types) {
#define PRINT_TENSOR_WITH_TYPE(type)                                        \
  std::vector<type> output_data(output_size);                               \
  output_tensor->CopyToCpu(output_data.data());                             \
  for (size_t j = 0; j < output_size && j < MAX_DISPLAY_OUTPUT_TENSOR_SIZE; \
       j++) {                                                               \
    std::cout << "[" << j << "] " << output_data[j] << std::endl;           \
  }
  auto output_names = predictor->GetOutputNames();
  for (int i = 0; i < output_types.size(); i++) {
    auto output_tensor = predictor->GetOutputHandle(output_names[i]);
    auto output_size = shape_production(output_tensor->shape());
    if (output_types[i] == "float32") {
      PRINT_TENSOR_WITH_TYPE(float)
    } else if (output_types[i] == "int32") {
      PRINT_TENSOR_WITH_TYPE(int32_t)
    } else if (output_types[i] == "int64") {
      PRINT_TENSOR_WITH_TYPE(int64_t)
    } else {
      printf(
          "Unsupported input data type '%s', only 'float32', 'int32', 'int64' "
          "are supported!\n",
          output_types[i].c_str());
      exit(-1);
    }
  }
#undef PRINT_TENSOR_WITH_TYPE
}

void check_output_tensors(const std::shared_ptr<Predictor> &tar_predictor,
                          const std::shared_ptr<Predictor> &ref_predictor,
                          const std::vector<std::string> &output_types) {
#define CHECK_TENSOR_WITH_TYPE(type)                                         \
  std::vector<type> tar_output_data(tar_output_size);                        \
  std::vector<type> ref_output_data(ref_output_size);                        \
  tar_output_tensor->CopyToCpu(tar_output_data.data());                      \
  ref_output_tensor->CopyToCpu(ref_output_data.data());                      \
  for (size_t j = 0; j < ref_output_size; j++) {                             \
    auto abs_diff = std::fabs(tar_output_data[j] - ref_output_data[j]);      \
    auto rel_diff = abs_diff / (std::fabs(ref_output_data[j]) + 1e-6);       \
    if (rel_diff < 0.01f) continue;                                          \
    std::cout << "val: " << tar_output_data[j]                               \
              << " ref: " << ref_output_data[j] << " abs_diff: " << abs_diff \
              << " rel_diff: " << rel_diff << std::endl;                     \
  }
  auto tar_output_names = tar_predictor->GetOutputNames();
  auto ref_output_names = ref_predictor->GetOutputNames();
  for (int i = 0; i < output_types.size(); i++) {
    auto tar_output_tensor =
        tar_predictor->GetOutputHandle(tar_output_names[i]);
    auto ref_output_tensor =
        ref_predictor->GetOutputHandle(ref_output_names[i]);
    auto tar_output_size = shape_production(tar_output_tensor->shape());
    auto ref_output_size = shape_production(ref_output_tensor->shape());
    if (tar_output_size != ref_output_size) {
      std::cout << "The size of output tensor[" << i << "] does not match."
                << std::endl;
      exit(-1);
    }
    if (output_types[i] == "float32") {
      CHECK_TENSOR_WITH_TYPE(float)
    } else if (output_types[i] == "int32") {
      CHECK_TENSOR_WITH_TYPE(int32_t)
    } else if (output_types[i] == "int64") {
      CHECK_TENSOR_WITH_TYPE(int64_t)
    }
  }
#undef CHECK_TENSOR_WITH_TYPE
}

void process(const std::string &model_dir,
             const std::vector<std::vector<int32_t>> &input_shapes,
             const std::vector<std::string> &input_types,
             const std::vector<std::string> &output_types,
             const std::string &device_name) {
  // Prepare for inference
  Config cfg;
  if (access(model_dir.c_str(), 0)) {
    printf("Invalid model dir!\n");
    return;
  }
  bool loaded = false;
  for (size_t i = 0; i < CANDIDATE_MODEL_FILES.size(); i++) {
    auto model_path = model_dir + "/" + CANDIDATE_MODEL_FILES[i].first;
    auto params_path = model_dir + "/" + CANDIDATE_MODEL_FILES[i].second;
    if (!access(model_path.c_str(), 0) && !access(params_path.c_str(), 0)) {
      cfg.SetModel(model_path, params_path);
      loaded = true;
      break;
    }
  }
  if (!loaded) {
    cfg.SetModel(model_dir);
  }
  if (device_name == "xpu") {
    cfg.EnableXpu();
  }
  // cfg.SwitchIrOptim(false);
  // cfg.pass_builder()->DeletePass("");
  cfg.EnableMemoryOptim();
  auto predictor = CreatePredictor(cfg);
  for (int i = 0; i < WARMUP_COUNT; i++) {
    fill_input_tensors(predictor, input_shapes, input_types, 1);
    predictor->Run();
  }
  double cur_cost = 0;
  double total_cost = 0;
  double max_cost = 0;
  double min_cost = std::numeric_limits<float>::max();
  for (int i = 0; i < REPEAT_COUNT; i++) {
    fill_input_tensors(predictor, input_shapes, input_types, 1);
    auto start = get_current_us();
    predictor->Run();
    auto end = get_current_us();
    cur_cost = (end - start) / 1000.0f;
    total_cost += cur_cost;
    if (cur_cost > max_cost) {
      max_cost = cur_cost;
    }
    if (cur_cost < min_cost) {
      min_cost = cur_cost;
    }
    printf("[%d] Prediction time: %f ms \n", i, cur_cost);
  }
  print_output_tensors(predictor, output_types);
  printf("Prediction time: avg %f ms, max %f ms, min %f ms\n",
         total_cost / REPEAT_COUNT,
         max_cost,
         min_cost);
  printf("Done.\n");
}

int main(int argc, char **argv) {
  if (argc < 6) {
    printf(
        "Usage: \n"
        "./demo model_dir input_shapes input_types output_types device_name");
    return -1;
  }
  std::string model_dir = argv[1];
  // Parsing the shape of input tensors from strings, supported formats:
  // "1,3,224,224" and "1,3,224,224:1,80"
  auto input_shapes = parse_shapes(argv[2]);
  // Parsing the data type of input and output tensors from strings, supported
  // formats: "float32" and "float32:int64:int8"
  auto input_types = parse_types(argv[3]);
  auto output_types = parse_types(argv[4]);
  std::string device_name = argv[5];
  // try {
  process(model_dir, input_shapes, input_types, output_types, device_name);
  //} catch (std::exception e) {
  // printf("An internal error occurred in PaddleInference.\n");
  // return -1;
  //}
  return 0;
}
