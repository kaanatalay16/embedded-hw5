/**
 * EE 4065 - Embedded Digital Image Processing
 * Homework 5 - Question 2: Handwritten Digit Recognition on STM32
 * 
 * This code demonstrates the deployment of a digit recognition model
 * on STM32 microcontroller using TensorFlow Lite for Microcontrollers.
 */

#include "main.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "mnist_model.h"

/* Private defines */
#define IMAGE_WIDTH         28
#define IMAGE_HEIGHT        28
#define IMAGE_SIZE          (IMAGE_WIDTH * IMAGE_HEIGHT)
#define NUM_CLASSES         10
#define TENSOR_ARENA_SIZE   (20 * 1024)

/* TensorFlow Lite variables */
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

/* Image buffer */
static uint8_t image_buffer[IMAGE_SIZE];

/* Function prototypes */
static void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_SPI_Init(void);
static void preprocess_image(uint8_t* raw_image, int8_t* processed);
static int run_inference(void);

/* Sample test images (from MNIST dataset) */
const uint8_t test_digit_7[IMAGE_SIZE] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,84,185,159,151,60,36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,222,254,254,254,254,241,198,198,198,198,198,198,198,198,170,52,0,0,0,0,0,0,
    0,0,0,0,0,0,67,114,72,114,163,227,254,225,254,254,254,250,229,254,254,140,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,17,66,14,67,67,67,59,21,236,254,106,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,83,253,209,18,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,233,255,83,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,254,238,44,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,59,249,254,62,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,133,254,187,5,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,205,248,58,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,126,254,182,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,75,251,240,57,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,19,221,254,166,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,3,203,254,219,35,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,38,254,254,77,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,31,224,254,115,1,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,133,254,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,61,242,254,254,52,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,121,254,254,219,40,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,121,254,207,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

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
    MX_SPI_Init();
    
    printf("Handwritten Digit Recognition on STM32\r\n");
    printf("=====================================\r\n");
    
    /* Load the TFLite model */
    const tflite::Model* model = tflite::GetModel(mnist_model);
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
    
    /* Test with sample digit */
    printf("\r\nTesting with sample digit (expected: 7)...\r\n");
    memcpy(image_buffer, test_digit_7, IMAGE_SIZE);
    
    int result = run_inference();
    printf("Predicted digit: %d\r\n", result);
    
    /* Main loop */
    printf("\r\nWaiting for image input...\r\n\n");
    while (1)
    {
        /* Check for button press or new image */
        if (HAL_GPIO_ReadPin(USER_BUTTON_GPIO_Port, USER_BUTTON_Pin) == GPIO_PIN_SET) {
            
            /* Capture image (implementation depends on camera/interface) */
            // capture_image(image_buffer);
            
            /* Run inference */
            int digit = run_inference();
            
            printf("Detected digit: %d\r\n", digit);
            
            /* Visual feedback */
            for (int i = 0; i <= digit; i++) {
                HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_SET);
                HAL_Delay(100);
                HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_RESET);
                HAL_Delay(100);
            }
            
            HAL_Delay(500);  // Debounce
        }
        
        HAL_Delay(10);
    }
}

/**
 * @brief Preprocess image for model input
 * @param raw_image: Raw 8-bit grayscale image
 * @param processed: Quantized output for model
 */
static void preprocess_image(uint8_t* raw_image, int8_t* processed)
{
    /* Get quantization parameters */
    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;
    
    /* Normalize to [0, 1] and quantize */
    for (int i = 0; i < IMAGE_SIZE; i++) {
        float normalized = raw_image[i] / 255.0f;
        int32_t quantized = (int32_t)(normalized / input_scale + input_zero_point);
        quantized = (quantized < -128) ? -128 : (quantized > 127) ? 127 : quantized;
        processed[i] = (int8_t)quantized;
    }
}

/**
 * @brief Run inference on the loaded model
 * @return Predicted digit (0-9)
 */
static int run_inference(void)
{
    /* Preprocess and copy image to input tensor */
    preprocess_image(image_buffer, input->data.int8);
    
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
    float max_prob = -1000.0f;
    
    printf("Probabilities: ");
    for (int i = 0; i < NUM_CLASSES; i++) {
        float prob = (output_data[i] - output_zero_point) * output_scale;
        printf("%d:%.2f ", i, prob);
        if (prob > max_prob) {
            max_prob = prob;
            max_idx = i;
        }
    }
    printf("\r\n");
    
    printf("Inference time: %lu ms, Confidence: %.2f%%\r\n", 
           inference_time, max_prob * 100);
    
    return max_idx;
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
 * @brief SPI Initialization for camera/display
 */
static void MX_SPI_Init(void)
{
    /* Initialize SPI for camera or display */
    // Implementation depends on specific peripherals
}

/* printf redirect to UART */
#ifdef __GNUC__
int _write(int file, char *ptr, int len)
{
    HAL_UART_Transmit(&huart2, (uint8_t*)ptr, len, HAL_MAX_DELAY);
    return len;
}
#endif

