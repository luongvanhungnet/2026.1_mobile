#include <jni.h>
#include <cstdint>
#include <sstream>
#include <vector>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/builtin_ops.h"

// An observing delegate. It does not replace, compute, or modify any graph node.
struct Probe {
  TfLiteDelegate delegate{};
  TfLiteDelegate* target{};
  bool prepared = false;
  int execution_nodes = 0, gpu_partitions = 0, gpu_original_nodes = 0;
  std::vector<int> cpu_codes;
};

static TfLiteStatus Prepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  auto* probe = static_cast<Probe*>(delegate->data_);
  if (probe->prepared) return kTfLiteOk;  // Fixed-shape models: retain the first before/after graph.
  TfLiteIntArray* plan = nullptr;
  if (context->GetExecutionPlan(context, &plan) != kTfLiteOk || !plan) return kTfLiteError;
  probe->execution_nodes = plan->size;
  probe->cpu_codes.clear();
  probe->gpu_partitions = probe->gpu_original_nodes = 0;
  for (int i = 0; i < plan->size; ++i) {
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, plan->data[i], &node, &registration) != kTfLiteOk) return kTfLiteError;
    if (registration->builtin_code == kTfLiteBuiltinDelegate && node->delegate == probe->target) {
      ++probe->gpu_partitions;
      auto* params = static_cast<TfLiteDelegateParams*>(node->builtin_data);
      if (params && params->nodes_to_replace) probe->gpu_original_nodes += params->nodes_to_replace->size;
    } else probe->cpu_codes.push_back(registration->builtin_code);
  }
  probe->prepared = true;
  return kTfLiteOk;
}

extern "C" JNIEXPORT jlong JNICALL
Java_vn_edu_gridrestore_inference_DelegateProbe_create(JNIEnv*, jobject, jlong target) {
  auto* probe = new Probe();
  probe->target = reinterpret_cast<TfLiteDelegate*>(target);
  probe->delegate.data_ = probe;
  probe->delegate.Prepare = Prepare;
  return reinterpret_cast<jlong>(probe);
}
extern "C" JNIEXPORT jlong JNICALL
Java_vn_edu_gridrestore_inference_DelegateProbe_handle(JNIEnv*, jobject, jlong pointer) {
  return reinterpret_cast<jlong>(&reinterpret_cast<Probe*>(pointer)->delegate);
}
extern "C" JNIEXPORT jstring JNICALL
Java_vn_edu_gridrestore_inference_DelegateProbe_snapshot(JNIEnv* env, jobject, jlong pointer) {
  auto* p = reinterpret_cast<Probe*>(pointer);
  std::ostringstream out;
  out << "{\"prepared\":" << (p->prepared ? "true" : "false")
      << ",\"executionNodes\":" << p->execution_nodes << ",\"gpuPartitions\":" << p->gpu_partitions
      << ",\"gpuOriginalNodes\":" << p->gpu_original_nodes << ",\"cpuNodes\":" << p->cpu_codes.size()
      << ",\"cpuBuiltinCodes\":[";
  for (size_t i = 0; i < p->cpu_codes.size(); ++i) { if (i) out << ','; out << p->cpu_codes[i]; }
  out << "]}";
  return env->NewStringUTF(out.str().c_str());
}
extern "C" JNIEXPORT void JNICALL
Java_vn_edu_gridrestore_inference_DelegateProbe_destroy(JNIEnv*, jobject, jlong pointer) {
  delete reinterpret_cast<Probe*>(pointer);
}
