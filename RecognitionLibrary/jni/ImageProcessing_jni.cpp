/*
 * ImageProcessing_jni.cpp
 *
 *  Created on: 23.05.2013
 *      Author: vchelban
 */

#include <ImageProcessing_jni.h>
#include <android/log.h>

#define LOG_TAG "FaceRecognition"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

#include <LibUtils.cpp>
#include <face/recognition/FaceRecognition.cpp>

#include <string>
#include <vector>

using namespace std;
using namespace cv;

//FaceDetectionProxy bindings
JNIEXPORT void JNICALL Java_com_endava_recognitionlibrary_proxy_FaceProcessingProxy_processImageAndDetectFaces(
		JNIEnv * jenv, jclass, jstring imageFilePath, jstring cascadeFilePath,
		jlong facesAddr, jint faceSize) {

	const char* jnamestr = jenv->GetStringUTFChars(cascadeFilePath, NULL);
	string stdCascadeFilePath(jnamestr);
	const char* imageStr = jenv->GetStringUTFChars(imageFilePath, NULL);
	string imageFileName(imageStr);
	try {
		LOGD("Face SIZE: %d", faceSize);
		vector<Rect> detectedFaces = detectFaces(imageFileName,
				stdCascadeFilePath, faceSize);
		vector_Rect_to_Mat(detectedFaces, *((Mat*) facesAddr));
	} catch (cv::Exception& e) {
		LOGD("processImageAndDetect caught cv::Exception: %s", e.what());

		jclass je = jenv->FindClass("org/opencv/core/CvException");
		if (!je)
			je = jenv->FindClass("java/lang/Exception");
		jenv->ThrowNew(je, e.what());

	} catch (...) {
		LOGD("processImageAndDetect caught unknown exception");
	}
}

JNIEXPORT bool JNICALL Java_com_endava_recognitionlibrary_proxy_FaceProcessingProxy_nativeCropFace(
		JNIEnv * jenv, jclass, jstring originalImagePath, jstring croppedPath,
		jint imgX, jint imgY, jint faceWidth, jint faceHeight) {
	bool result = false;
	try {
		const char* jpathstr = jenv->GetStringUTFChars(croppedPath, NULL);
		string stdFinalPath(jpathstr);

		const char* jorigpathstr = jenv->GetStringUTFChars(originalImagePath,
				NULL);
		string stdOrigPath(jorigpathstr);

		Mat loadedImg = imread(stdOrigPath);

		Mat croppedImage = cropImage(loadedImg,
				prepareFaceForCropping(imgX, imgY, faceWidth, faceHeight));
		Mat scalledImage = scaleImage(croppedImage, FACE_FRAME_SIZE,
				FACE_FRAME_SIZE);
		result = saveImage(scalledImage, stdFinalPath);
	} catch (cv::Exception& e) {
		LOGD("nativeCropFace caught cv::Exception: %s", e.what());
	} catch (...) {
		LOGD("nativeCropFace caught unknown exception");
	}
	return result;
}

JNIEXPORT bool JNICALL Java_com_endava_recognitionlibrary_proxy_FaceProcessingProxy_rotateImage(
		JNIEnv * jenv, jclass, jstring originalImgPath,
		jdouble rotationDegree) {
	bool result = false;
	try {
		const char* jpathstr = jenv->GetStringUTFChars(originalImgPath, NULL);
		string stdFinalPath(jpathstr);

		Mat loadedImg = imread(stdFinalPath);
		double degreeToRotate = 360 - rotationDegree;
		Mat rotatedImg;
		if(degreeToRotate == 90 || degreeToRotate == 270){
			Mat rotatedImg = Mat::zeros(loadedImg.cols, loadedImg.rows,
					loadedImg.type());
		}
		else{
		Mat rotatedImg = Mat::zeros(loadedImg.rows, loadedImg.cols,
				loadedImg.type());
		}
		rotate(loadedImg, degreeToRotate, rotatedImg);
		result = saveImage(rotatedImg, stdFinalPath);
	} catch (cv::Exception& e) {
		LOGD("rotateImage caught cv::Exception: %s", e.what());
	} catch (...) {
		LOGD("rotateImage caught unknown exception");
	}
	return result;
}

//FaceRecognitionProxy bindings
JNIEXPORT jlong JNICALL Java_com_endava_recognitionlibrary_proxy_FaceRecognitionProxy_createRecognizer(
		JNIEnv * jenv, jclass, jstring modelName, int recognitionModelOrdinal) {
	long result = 0;
	try {
		const char* jnamestr = jenv->GetStringUTFChars(modelName, NULL);
		string stdModelName(jnamestr);
		result = (long) new FaceRecognition(stdModelName,
				recognitionModelOrdinal);
	} catch (cv::Exception& e) {
		LOGD("createRecognizer caught cv::Exception: %s", e.what());
	} catch (...) {
		LOGD("createRecognizer caught unknown exception");
	}

	return result;
}
JNIEXPORT jlong JNICALL Java_com_endava_recognitionlibrary_proxy_FaceRecognitionProxy_createRecognizerWithCascade(
		JNIEnv * jenv, jclass, jstring modelName, jstring cascadeFilePath,
		jint recognitionModelOrdinal) {
	long result = 0;
	try {
		const char* jnamestr = jenv->GetStringUTFChars(modelName, NULL);
		string stdModelPath(jnamestr);
		const char* jCascadeStr = jenv->GetStringUTFChars(cascadeFilePath,
				NULL);
		string stdCascadePath(jCascadeStr);
		result = (long) new FaceRecognition(stdModelPath, stdCascadePath,
				recognitionModelOrdinal);
	} catch (cv::Exception& e) {
		LOGD("createRecognizer caught cv::Exception: %s", e.what());
	} catch (...) {
		LOGD("createRecognizer caught unknown exception");
	}

	return result;
}

JNIEXPORT void JNICALL Java_com_endava_recognitionlibrary_proxy_FaceRecognitionProxy_releaseRecognizer(
		JNIEnv * jenv, jclass, jlong thiz) {
	try {
		if (thiz != 0) {
			delete (FaceRecognition*) thiz;
		}
	} catch (cv::Exception& e) {
		LOGD("releaseRecognizer caught cv::Exception: %s", e.what());
	} catch (...) {
		LOGD("releaseRecognizer caught unknown exception");
	}
}

JNIEXPORT jboolean JNICALL Java_com_endava_recognitionlibrary_proxy_FaceRecognitionProxy_predictFace(
		JNIEnv * jenv, jclass, jlong thiz, jstring imagePath,
		jstring processedImgPath, jint contactID, jdouble rotationDegree) {
	bool result = false;
	try {
		const char* jnamestr = jenv->GetStringUTFChars(imagePath, NULL);
		string stdImagePath(jnamestr);

		const char* processedImgStr = jenv->GetStringUTFChars(processedImgPath,
				NULL);
		string stdProcessedImagePath(processedImgStr);
		if (thiz != 0) {
			result = ((FaceRecognition*) thiz)->predictFace(stdImagePath,
					stdProcessedImagePath, contactID, rotationDegree);
		}
	} catch (cv::Exception& e) {
		LOGD("predictFace caught cv::Exception: %s", e.what());
	} catch (...) {
		LOGD("predictFace caught unknown exception");
	}
	return result;
}

JNIEXPORT jboolean JNICALL Java_com_endava_recognitionlibrary_proxy_FaceRecognitionProxy_createModel(
		JNIEnv * jenv, jclass, jlong thiz, jstring modelName,
		jobjectArray sourceImagesArr, jintArray labelIDs) {
	bool result = false;
	if (thiz != 0) {
		try {
			vector<string> images = vector<string>();

			const char* jnamestr = jenv->GetStringUTFChars(modelName, NULL);
			string stdModelName(jnamestr);
			for (int i = 0; i < jenv->GetArrayLength(sourceImagesArr); i++) {
				jstring jstr = (jstring) jenv->GetObjectArrayElement(
						sourceImagesArr, i);

				const char* jimagestr = jenv->GetStringUTFChars(jstr, NULL);
				string stdImagePathStr(jimagestr);
				images.push_back(stdImagePathStr);
			}
			//converting labe IDs to vector
			jboolean isCopy = true;
			int labelLength = jenv->GetArrayLength(labelIDs);
			int* ints = jenv->GetIntArrayElements(labelIDs, &isCopy);

			vector<int> labels = vector<int>();
			for (int i = 0; i < labelLength; i++) {
				labels.push_back(ints[i]);
			}

			result = ((FaceRecognition*) thiz)->createModel(stdModelName, images,
					labels);
		} catch (cv::Exception& e) {
			LOGD("createModel caught cv::Exception: %s", e.what());
		} catch (...) {
			LOGD("createModel caught unknown exception");
		}
	}
	return result;
}

JNIEXPORT void JNICALL Java_com_endava_recognitionlibrary_proxy_FaceRecognitionProxy_updateModel(
		JNIEnv * jenv, jclass, jlong thiz, jstring modelName,
		jobjectArray sourceImagesArr, jintArray labelIDs) {
	if (thiz != 0) {
		try {
			vector<string> images = vector<string>();

			const char* jnamestr = jenv->GetStringUTFChars(modelName, NULL);
			string stdModelName(jnamestr);
			for (int i = 0; i < jenv->GetArrayLength(sourceImagesArr); i++) {
				jstring jstr = (jstring) jenv->GetObjectArrayElement(
						sourceImagesArr, i);

				const char* jimagestr = jenv->GetStringUTFChars(jstr, NULL);
				string stdImagePathStr(jimagestr);
				images.push_back(stdImagePathStr);
			}

			//converting labe IDs to vector
			jboolean isCopy = true;
			int labelLength = jenv->GetArrayLength(labelIDs);
			int* ints = jenv->GetIntArrayElements(labelIDs, &isCopy);

			vector<int> labels = vector<int>();
			for (int i = 0; i < labelLength; i++) {
				labels.push_back(ints[i]);
			}

			((FaceRecognition*) thiz)->updateModel(stdModelName, images,
					labels);
		} catch (cv::Exception& e) {
			LOGD("updateModel caught cv::Exception: %s", e.what());
		} catch (...) {
			LOGD("updateModel caught unknown exception");
		}
	}
}
