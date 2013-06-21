/*
 * LibUtils.h
 *
 *  Created on: 05.06.2013
 *      Author: vchelban
 */

#ifndef LIBUTILS_H_
#define LIBUTILS_H_

using namespace cv;

/**
 * Scale an image to a specified size using cubic interpolation.
 */
Mat scaleImage(Mat originalImage, int imageWidth, int imageHeight);

/**
 * Crop a region of interest (ROI) from an image.
 */
Mat cropImage(Mat originalImage, const Rect & roi);

/**
 * Prepares the face to be cropped from original detected Rect.
 */
Rect prepareFaceForCropping(int imgX, int imgY, int faceWidth, int faceHeight);

/**
 * Saves the specified image data to the specified path using PNG compression.
 */
bool saveImage(Mat sourceImage, string& imagePath);

/**
 * Loads an image and tries to detect any faces using the specified CascadeClassifier.
 */
vector<Rect> detectFaces(string& imageFilePath, string& cascadeFilePath, int minFaceSize);

/**
 * Loads an image and tries to detect any faces using the specified CascadeClassifier.
 */
vector<Rect> detectFaces(Mat& loadedImageData, string& cascadeFilePath, int minFaceSize);


/**
 * Converts a vector of Rect to a MatOfRect.
 */
void vector_Rect_to_Mat(vector<Rect>& v_rect, Mat& facesAddr);


void rotate(Mat& src, double degree, Mat& dst);

#endif /* LIBUTILS_H_ */
