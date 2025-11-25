

#include <iostream>
#include <memory>


#include <"tensorflow/lite/interpreter.h>
#include <"tensorflow/lite/kernels/register.h>
#include <"tensorflow/lite/model.h>


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
		return 1;
	}

	// Build interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	std::unique_ptr<tflite::Interpreter> interpreter;
	tflite::InterpreterBuilder(*model, resolver)( &interpreter );
	if (!interpreter) {
		std::cerr << "Failed to construct interpreter\n";
		return 1;
	}

	// Allocate tensors
	if (interpreter->AllocateTensors() != kTfLiteOk) {
		std::cerr << "Failed to allocate tensors\n";
		return 1;
	}

	//int input_count = interpreter->inputs().size();
	//std::cout << "Number of input tensors: " << input_count << std::endl;
	// Number of input tensors: 1
	
	
	int input_index = interpreter->inputs()[0];
	TfLiteTensor* input_tensor = interpreter->tensor(input_index);
	//for (int i = 0; i < input_tensor->dims->size; ++i) {
	//    std::cout << input_tensor->dims->data[i] << " ";
	//}
	//std::cout << std::endl;
	// 1 224 224 3
	// Check tensor type
	if (input_tensor->type != kTfLiteUInt8) {
		std::cerr << "Expected input tensor type kTfLiteUInt8\n";
		return false;
	}
	
	
	cv::Mat img = cv::imread(image_path);
	if (img.empty()) {
		std::cerr << "Failed to load image: " << image_path << std::endl;
		return false;
	}
	
	cv::resize(img, img, cv::Size(224, 224));
	
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
	
	// Copy image data to tensor
	std::memcpy(input_tensor->data.uint8, img.data, 224 * 224 * 3);
	
	
	if (interpreter->Invoke() != kTfLiteOk) {
		std::cerr << "Failed to invoke interpreter\n";
		return 1;
	}

	int output_index = interpreter->outputs()[0];
	TfLiteTensor* output_tensor = interpreter->tensor(output_index);
	
	//for (int i = 0; i < output_tensor->dims->size; ++i) {
	//	std::cout << output_tensor->dims->data[i] << " ";
	//}
	//std::cout << std::endl;
	// 1 1001
	
	// Dequantize and store scores with class indices
	std::vector<std::pair<int, float>> scores;
	for (int i = 0; i < 1001; ++i) {
		uint8_t value = output_tensor->data.uint8[i];
		float score = output_tensor->params.scale * (value - output_tensor->params.zero_point);
		scores.emplace_back(i, score);
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

1 1001 Class 653: score = 0.78125
Class 907: score = 0.105469
Class 458: score = 0.0195312
Class 668: score = 0.015625
Class 466: score = 0.0117188
Model loaded and interpreter ready!


*/


