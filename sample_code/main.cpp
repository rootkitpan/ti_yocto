#include <iostream>
#include <memory>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

#include <opencv2/opencv.hpp>


/*

g++ main.cpp -o run_model /usr/lib/libtensorflow-lite.a -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

./run_model

*/


const char* model_path = "/usr/share/tensorflow-lite/examples/mobilenet_v1_1.0_224_quant.tflite";
const char* image_path = "/usr/share/tensorflow-lite/examples/grace_hopper.bmp";
const char* label_path = "/usr/share/tensorflow-lite/examples/labels.txt";



int main(int argc, char* argv[])
{
	tflite::StderrReporter error_reporter;

	// Load model
	// std::unique_ptr< FlatBufferModel >
	auto model = tflite::FlatBufferModel::BuildFromFile( model_path, &error_reporter );
	if (!model) {
		return -1;
	}

	// Build interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	std::unique_ptr<tflite::Interpreter> interpreter;
	tflite::InterpreterBuilder(*model, resolver)( &interpreter );
	if (!interpreter) {
		std::cerr << "Failed to construct interpreter\n";
		return -1;
	}

	// Optional: AM3358 is single-core, 1 thread is fine
	interpreter->SetNumThreads(1);
	
	
	// Allocate tensors
	if (interpreter->AllocateTensors() != kTfLiteOk) {
		std::cerr << "Failed to allocate tensors\n";
		return -1;
	}
	
	
	// Check Input Tensor, should be [1, 224, 224, 3]
	int input_index = interpreter->inputs()[0];
	TfLiteTensor* input_tensor = interpreter->tensor(input_index);
	const TfLiteIntArray* input_tensor_dims = input_tensor->dims;
	if(
		(input_tensor_dims->size != 4)
		|| (input_tensor_dims->data[1] != 224)
		|| (input_tensor_dims->data[2] != 224)
		|| (input_tensor_dims->data[3] != 3)
	){
		std::cerr << "Input shape error\n";
		return -1;
	}

	// Check tensor type
	if (input_tensor->type != kTfLiteUInt8) {
		std::cerr << "Expected input tensor type kTfLiteUInt8\n";
		return -1;
	}
	
	
	// use opencv to load image
	cv::Mat img = cv::imread(image_path);
	if (img.empty()) {
		std::cerr << "Failed to load image: " << image_path << std::endl;
		return -1;
	}
	// change image to RGB to fit model input
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
	// resize image to 224*224 to fit model input
	cv::Mat resized_img;
	cv::resize(img, resized_img, cv::Size(224, 224));
	
	// Copy image data to tensor
	std::memcpy(
		input_tensor->data.uint8,
		resized_img.data,
		resized_img.total() * resized_img.elemSize()		// equals to 224 * 224 * 3
	);
	
	
	if (interpreter->Invoke() != kTfLiteOk) {
		std::cerr << "Failed to invoke interpreter\n";
		return -1;
	}

	
	
	// Get output
	int output_index = interpreter->outputs()[0];
	TfLiteTensor* output_tensor = interpreter->tensor(output_index);
	// print out output tensor shape, should be [1, 1001]
	for (int i = 0; i < output_tensor->dims->size; ++i) {
		std::cout << output_tensor->dims->data[i] << " ";
	}
	std::cout << std::endl;
	// output tensor type should be uint8
	if( output_tensor->type !=  kTfLiteUInt8 ){
		std::cout << "Output tensor shape is not kTfLiteUInt8" << std::endl;
		return -1;
	}
	
	
	
	
	// Dequantize and store scores with class indices
	std::vector<std::pair<int, float>> scores;
	for (int i = 0; i < 1001; ++i) {
		uint8_t value = output_tensor->data.uint8[i];
		float scale = output_tensor->params.scale;
		
#if 0
		if(value > 0)
			value--;
		float scale = 1.0/255;
#endif
		int zero_point = output_tensor->params.zero_point;
		float score = scale * (value - zero_point);
		scores.emplace_back(i, score);
		
		if(
			( i == 653 )
			|| ( i == 907 )
			|| ( i == 458 )
			|| ( i == 668 )
			|| ( i == 466 )
		){
			std::cout << "[" << i << "]:"
					<< "[" << static_cast<int>(value) << "]"
					<< "[" << scale << "]"
					<< "[" << static_cast<int>(zero_point) << "]"
					<< "[" << score << "]"
					<< std::endl;
		}
	}

	// Sort by score descending
	std::sort(
		scores.begin(),
		scores.end(),
		[](const std::pair<int, float>& a, const std::pair<int, float>& b) {
			return a.second > b.second;
		}
	);

	// Print top 5
	for (int i = 0; i < 5; ++i) {
	    std::cout << "Class " << scores[i].first << ": score = " << scores[i].second << std::endl;
	}
	
	
	
	
	
	return 0;
}




/*
[Mine]
Class 653: score = 0.78125
Class 907: score = 0.105469
Class 458: score = 0.0195312
Class 668: score = 0.015625
Class 466: score = 0.0117188

value-- and scale = 1/255
Class 653: score = 0.780392
Class 907: score = 0.101961
Class 458: score = 0.0156863
Class 668: score = 0.0117647
Class 466: score = 0.00784314


[example]
0.780392: 653 military uniform
0.105882: 907 Windsor tie
0.0156863: 458 bow tie
0.0117647: 466 bulletproof vest
0.00784314: 835 suit



*/


