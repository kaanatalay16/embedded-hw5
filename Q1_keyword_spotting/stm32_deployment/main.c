/**
 * EE 4065 - Embedded Digital Image Processing
 * Homework 5 - Question 1: Keyword Spotting on STM32
 * 
 * This code demonstrates the deployment of a keyword spotting model
 * on STM32 microcontroller using TensorFlow Lite for Microcontrollers.
 */

#include "main.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "kws_model.h"

/* Private defines */
#define SAMPLE_RATE         16000
#define AUDIO_LENGTH        16000  // 1 second of audio
#define MFCC_FRAME_LEN      400
#define MFCC_FRAME_STEP     160
#define MFCC_NUM_BINS       40
#define MFCC_NUM_FRAMES     98
#define TENSOR_ARENA_SIZE   (50 * 1024)

/* Keywords */
const char* keywords[] = {
    "yes", "no", "up", "down", "left", 
    "right", "on", "off", "stop", "go"
};
const int num_keywords = 10;

/* TensorFlow Lite variables */
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

/* Audio buffer */
static int16_t audio_buffer[AUDIO_LENGTH];
static float mfcc_features[MFCC_NUM_FRAMES * MFCC_NUM_BINS];

/* Function prototypes */
static void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_I2S_Init(void);
static void compute_mfcc(int16_t* audio, float* mfcc_out);
static int run_inference(void);

/**
 * @brief Application entry point
 */
int main(void)
{
    /* MCU Configuration */
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART2_UART_Init();
    MX_I2S_Init();
    
    printf("Keyword Spotting on STM32\r\n");
    printf("========================\r\n");
    
    /* Load the TFLite model */
    const tflite::Model* model = tflite::GetModel(kws_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema version mismatch!\r\n");
        while(1);
    }
    
    /* Create op resolver with required operations */
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();
    
    /* Build interpreter */
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;
    
    /* Allocate tensors */
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("Failed to allocate tensors!\r\n");
        while(1);
    }
    
    /* Get input and output tensors */
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    printf("Model loaded successfully!\r\n");
    printf("Input shape: [%d, %d, %d, %d]\r\n", 
           input->dims->data[0], input->dims->data[1],
           input->dims->data[2], input->dims->data[3]);
    printf("Arena used: %d bytes\r\n", 
           interpreter->arena_used_bytes());
    printf("\r\nListening for keywords...\r\n\n");
    
    /* Main loop */
    while (1)
    {
        /* Capture audio (implementation depends on hardware) */
        // capture_audio(audio_buffer, AUDIO_LENGTH);
        
        /* Compute MFCC features */
        compute_mfcc(audio_buffer, mfcc_features);
        
        /* Run inference */
        int keyword_id = run_inference();
        
        if (keyword_id >= 0 && keyword_id < num_keywords) {
            printf("Detected: %s\r\n", keywords[keyword_id]);
            
            /* Visual feedback with LED */
            HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_SET);
            HAL_Delay(200);
            HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_RESET);
        }
        
        HAL_Delay(100);  // Small delay between inferences
    }
}

/**
 * @brief Compute MFCC features from audio samples
 * @param audio: Input audio buffer
 * @param mfcc_out: Output MFCC features
 */
static void compute_mfcc(int16_t* audio, float* mfcc_out)
{
    /* 
     * MFCC computation steps:
     * 1. Pre-emphasis
     * 2. Framing
     * 3. Windowing (Hamming)
     * 4. FFT
     * 5. Mel filterbank
     * 6. Log compression
     * 7. DCT
     * 
     * Note: This is a simplified placeholder.
     * For production, use optimized DSP libraries like CMSIS-DSP.
     */
    
    // Pre-emphasis coefficient
    const float pre_emphasis = 0.97f;
    
    // Apply pre-emphasis
    float emphasized[AUDIO_LENGTH];
    emphasized[0] = (float)audio[0];
    for (int i = 1; i < AUDIO_LENGTH; i++) {
        emphasized[i] = (float)audio[i] - pre_emphasis * (float)audio[i-1];
    }
    
    // Frame processing
    int frame_idx = 0;
    for (int start = 0; start < AUDIO_LENGTH - MFCC_FRAME_LEN && 
         frame_idx < MFCC_NUM_FRAMES; start += MFCC_FRAME_STEP) {
        
        // Apply Hamming window and compute frame features
        float frame[MFCC_FRAME_LEN];
        for (int i = 0; i < MFCC_FRAME_LEN; i++) {
            float window = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (MFCC_FRAME_LEN - 1));
            frame[i] = emphasized[start + i] * window;
        }
        
        // Placeholder: In real implementation, compute FFT, mel filterbank, 
        // log compression, and DCT here using CMSIS-DSP
        
        // Store MFCC coefficients for this frame
        for (int i = 0; i < MFCC_NUM_BINS; i++) {
            mfcc_out[frame_idx * MFCC_NUM_BINS + i] = 0.0f;  // Placeholder
        }
        
        frame_idx++;
    }
}

/**
 * @brief Run inference on the loaded model
 * @return Index of detected keyword, or -1 if no keyword detected
 */
static int run_inference(void)
{
    /* Get quantization parameters */
    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;
    
    /* Quantize and copy MFCC features to input tensor */
    int8_t* input_data = input->data.int8;
    for (int i = 0; i < MFCC_NUM_FRAMES * MFCC_NUM_BINS; i++) {
        int32_t quantized = (int32_t)(mfcc_features[i] / input_scale + input_zero_point);
        quantized = (quantized < -128) ? -128 : (quantized > 127) ? 127 : quantized;
        input_data[i] = (int8_t)quantized;
    }
    
    /* Run inference */
    uint32_t start_time = HAL_GetTick();
    TfLiteStatus status = interpreter->Invoke();
    uint32_t inference_time = HAL_GetTick() - start_time;
    
    if (status != kTfLiteOk) {
        printf("Inference failed!\r\n");
        return -1;
    }
    
    /* Get output and find max probability */
    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;
    int8_t* output_data = output->data.int8;
    
    int max_idx = 0;
    float max_prob = 0.0f;
    
    for (int i = 0; i < num_keywords; i++) {
        float prob = (output_data[i] - output_zero_point) * output_scale;
        if (prob > max_prob) {
            max_prob = prob;
            max_idx = i;
        }
    }
    
    /* Only return keyword if confidence is above threshold */
    const float threshold = 0.7f;
    if (max_prob >= threshold) {
        printf("Inference time: %lu ms, Confidence: %.2f%%\r\n", 
               inference_time, max_prob * 100);
        return max_idx;
    }
    
    return -1;  // No keyword detected with sufficient confidence
}

/**
 * @brief System Clock Configuration
 */
static void SystemClock_Config(void)
{
    /* Configure system clock for maximum performance */
    // Implementation depends on specific STM32 variant
}

/**
 * @brief GPIO Initialization
 */
static void MX_GPIO_Init(void)
{
    /* Initialize LED and button GPIOs */
    // Implementation depends on specific board
}

/**
 * @brief UART Initialization for debug output
 */
static void MX_USART2_UART_Init(void)
{
    /* Initialize UART for printf */
    // Implementation depends on specific STM32 variant
}

/**
 * @brief I2S Initialization for audio capture
 */
static void MX_I2S_Init(void)
{
    /* Initialize I2S for microphone input */
    // Implementation depends on specific microphone and board
}

/* printf redirect to UART */
#ifdef __GNUC__
int _write(int file, char *ptr, int len)
{
    HAL_UART_Transmit(&huart2, (uint8_t*)ptr, len, HAL_MAX_DELAY);
    return len;
}
#endif

