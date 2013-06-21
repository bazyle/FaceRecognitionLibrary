/*
 * FaceDetector.cpp
 *
 *  Created on: 24.05.2013
 *      Author: vchelban
 */

#include "FaceRecognition.h"
#include "opencv2/highgui/highgui.hpp"
#include "vector"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

FaceRecognition::FaceRecognition(string& modelPath, int recognitionMethod) :
		modelPath(modelPath), recognitionMethod(recognitionMethod) {
	initRecognitionModel();
	loadSavedModel();
}

FaceRecognition::FaceRecognition(string& modelPath, string& cascadeFilePath,
		int recognitionMethod) :
		modelPath(modelPath), cascadeFilePath(cascadeFilePath), recognitionMethod(
				recognitionMethod) {
	initRecognitionModel();
	loadSavedModel();
}

FaceRecognition::~FaceRecognition() {
}

void FaceRecognition::initRecognitionModel() {
	switch (recognitionMethod) {
	case EIGEN_METHOD: {
		recognizer = createEigenFaceRecognizer(80, DBL_MAX);
		break;
	}
	case FISHER_FACES_METHOD: {
		recognizer = createFisherFaceRecognizer();
		break;
	}
	case LBPH: {
		recognizer = createLBPHFaceRecognizer();
		break;
	}
	default:
		recognizer = createLBPHFaceRecognizer();
		break;
	}
}

void FaceRecognition::loadSavedModel() {
	if (!modelPath.empty()) {
		recognizer->load(modelPath);
	}
}

bool FaceRecognition::predictFace(string& originalImageName,
		string& processedImgPath, int contactID, double rotationDegree) {
	bool recognized = false;
	//load the original image

	Mat imgGray = imread(originalImageName, CV_LOAD_IMAGE_GRAYSCALE);

	Mat imgToProcess = Mat::zeros(imgGray.rows, imgGray.cols, imgGray.type());

	if (rotationDegree != 0) {
		rotate(imgGray, 360 - rotationDegree, imgToProcess);
	} else {
		imgToProcess = imgGray;
	}

	//cvtColor(originalImg, imgGray, CV_BGR2GRAY);

	vector<Rect> detectedFaces;
	detectedFaces = detectFaces(imgToProcess, cascadeFilePath);
	int predicted_label = -1;
	double predicted_confidence, confidence = 41.0;

	if (detectedFaces.size() > 0) {
		Mat originalImg = imread(originalImageName, CV_LOAD_IMAGE_UNCHANGED);

		for (std::vector<Rect>::size_type i = 0;
				i < detectedFaces.size() && !recognized; i++) {
			Rect face = detectedFaces[i];
			Mat croppedFace = cropImage(imgToProcess,
					prepareFaceForCropping(face.x, face.y, face.width,
							face.height));
			Mat scalledFace = scaleImage(croppedFace, FACE_FRAME_SIZE,
					FACE_FRAME_SIZE);
			predicted_label = -1;

			// Get the prediction and associated confidence from the model
			recognizer->predict(scalledFace, predicted_label,
					predicted_confidence);
			//the person was recognized!
			if (predicted_label == contactID
					&& predicted_confidence <= confidence) {

				/*
				 if(originalImg.rows > originalImg.cols){
				 width = (originalImg.cols/originalImg.rows) * RECOGNIZED_THUMB_SIZE;
				 }
				 else{
				 height =(originalImg.rows / originalImg.cols) * RECOGNIZED_THUMB_SIZE;
				 }
				 */
				Mat scalledImage = scaleImage(originalImg,
						RECOGNIZED_THUMB_SIZE, RECOGNIZED_THUMB_SIZE);
				Mat finalImage = Mat::zeros(scalledImage.rows,
						scalledImage.cols, scalledImage.type());
				if (rotationDegree != 0) {
					rotate(scalledImage, 360 - rotationDegree, finalImage);
				} else {
					finalImage = scalledImage;
				}

				if (saveImage(finalImage, processedImgPath)) {
					recognized = true;
				}
			}
		}
	}

	LOGD("Predicted confidence: %f", predicted_confidence);
	return recognized;
}

void FaceRecognition::createModel(string& modelPath,
		vector<string> sourceImages, vector<int> labelIDs) {
	LOGD("INPUT DATA SIZES: images=%d, labels=%d", sourceImages.size(), labelIDs.size());

	if (recognizer != NULL && sourceImages.size() == labelIDs.size()) {
		vector<Mat> images;

		//vector<int> labels;
		for (std::vector<string>::size_type i = 0; i < sourceImages.size();
				i++) {
			string& imagePath = sourceImages[i];
			images.push_back(imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE));

			//if(i == (sourceImages.size() - 1)){
			//labels.push_back(234);
			//}
			///else{
			//labels.push_back(labelIDs);
			//}
		}
		if (images.size() >= 2) {
			recognizer->train(images, labelIDs);
			recognizer->save(modelPath);
		}
	}
}

void FaceRecognition::updateModel(string& modelPath,
		vector<string> sourceImages, vector<int> labelIDs) {
	if (recognizer != NULL && sourceImages.size() == labelIDs.size()) {
		vector<Mat> images;

		//vector<int> labels;
		for (std::vector<string>::size_type i = 0; i < sourceImages.size();
				i++) {
			string& imagePath = sourceImages[i];
			images.push_back(imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE));
			//labels.push_back(labelIDs);
		}


		if (images.size() > 2) {
			recognizer->update(images, labelIDs);
			recognizer->save(modelPath);
		}
	}
}

