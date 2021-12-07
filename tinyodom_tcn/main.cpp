#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "model.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "mbed.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
Timer t;
//Window Size and Number of channels to use
const int numSamples = 400;
int samplesRead = 0;
const int numChannels = 6;
// Globals, used for compatibility with Arduino-style sketches.
namespace {
    tflite::ErrorReporter* error_reporter = nullptr;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    int inference_count = 0;
    tflite::MicroProfiler* profiler = nullptr;
constexpr int kTensorArenaSize=40*1000;
    uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

int main(int argc, char* argv[]) {
    uint32_t failures = 0;
    tflite::InitializeTarget();
    // Set up logging. Google style is to avoid globals or statics because of
    // lifetime uncertainty, but since this has a trivial destructor it's okay.
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;
    static tflite::MicroProfiler micro_profiler_reporter;
    profiler = &micro_profiler_reporter;
    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("Invalid schema version\n");
    }
    // This pulls in all the operation 
  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;
 // Build an interpreter to run the model with.
 static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter,profiler);
    interpreter = &static_interpreter;
    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
     printf("Failed to allocate tensors\n");
    }
    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);
        while(samplesRead<numSamples){
            for (int i = 0; i<numChannels; i++){
input->data.f[samplesRead * numChannels + i] = ((float)rand()/(float)(RAND_MAX))*5.0;
            }
        samplesRead++;
        }
        samplesRead = 0;
        t.start();
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            printf("Failed to invoke\n");
        }
        t.stop();
        for (int i = 0; i < 100; i++){
        	printf("timer output: %f\n", t.read());
        }
        t.reset();
    }
