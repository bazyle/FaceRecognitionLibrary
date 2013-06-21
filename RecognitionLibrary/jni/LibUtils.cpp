/*
 * LibUtils.cpp
 *
 *  Created on: 05.06.2013
 *      Author: vchelban
 */

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <LibUtils.h>



using namespace cv;

const int FACE_FRAME_SIZE = 100;
const int RECOGNIZED_THUMB_SIZE = 150;
const int FACE_MARGIN_RATIO = 15;

/**
 * Scale an image to a specified size using cubic interpolation.
 */
Mat scaleImage(Mat originalImage, int imageWidth, int imageHeight) {
	Size imageSize = cvSize(imageWidth, imageHeight);
	Mat scalledImage(imageSize, originalImage.type());
	resize(originalImage, scalledImage, imageSize, 0, 0, INTER_CUBIC);
	return scalledImage;
}

/**
 * Crop a region of interest (ROI) from an image.
 */
Mat cropImage(Mat originalImage, const Rect & roi) {
	Mat croppedImage(originalImage, roi);
	return croppedImage;
}

Rect prepareFaceForCropping(int imgX, int imgY, int faceWidth, int faceHeight){
	int faceMarginForCropWidth = (FACE_MARGIN_RATIO * faceWidth) / 100;
	int faceMarginForCropHeight = (FACE_MARGIN_RATIO * faceHeight) / 100;
	int faceX = imgX + faceMarginForCropWidth;
	int faceY = imgY + faceMarginForCropHeight;
	int croppedFaceWidth = faceWidth - faceMarginForCropWidth * 2;
	int croppedFaceHeight = faceHeight - faceMarginForCropHeight * 2;
	return cvRect(faceX, faceY, croppedFaceWidth, croppedFaceHeight);
}

/**
 * Saves the specified image data to the specified path using PNG compression.
 */
bool saveImage(Mat sourceImage, string& imagePath) {
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	return imwrite(imagePath, sourceImage, compression_params);
}

/**
 * Loads an image and tries to detect any faces using the specified CascadeClassifier.
 */
vector<Rect> detectFaces(Mat& loadedImageData, string& cascadeFilePath, int minFaceSize = 5) {
	CascadeClassifier face_cascade;
	vector<Rect> rectFaces;
	if (face_cascade.load(cascadeFilePath)) {
		int height = loadedImageData.rows;
		int mAbsoluteFaceSize = 0;
		//set the face ratio
		double mRelativeFaceSize = (double) (minFaceSize * 0.01);
		int maxFaceSize = round((double) height * 0.08);

		if (round(height * mRelativeFaceSize) > 0) {
			mAbsoluteFaceSize = round((double) height * mRelativeFaceSize);
		}
		Mat frame_gray;
		//cvtColor(loadedImageData, frame_gray, CV_BGR2GRAY);
		//equalizeHist(loadedImageData, frame_gray);

		face_cascade.detectMultiScale(loadedImageData, rectFaces, 1.1, 2, 2,
				Size(mAbsoluteFaceSize, mAbsoluteFaceSize), Size());
	}
	return rectFaces;
}


vector<Rect> detectFaces(string& imageFilePath, string& cascadeFilePath, int minFaceSize = 5) {
	Mat loadedImageData = imread(imageFilePath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	vector<Rect> faces = detectFaces(loadedImageData, cascadeFilePath, minFaceSize);
	return faces;
}

void vector_Rect_to_Mat(vector<Rect>& v_rect, Mat& facesAddr) {
	facesAddr = Mat(v_rect, true);
}

void rotate(Mat& src, double degree, Mat& dst){
	Point2f center(src.cols/2, src.rows/2);
	Mat rotMat = getRotationMatrix2D(center, degree, 1.0);
	warpAffine(src, dst, rotMat, cvSize(src.rows, src.cols));
}


