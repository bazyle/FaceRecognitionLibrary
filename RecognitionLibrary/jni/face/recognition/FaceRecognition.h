/*
 * FaceDetector.h
 *
 *  Created on: 24.05.2013
 *      Author: vchelban
 */



#ifndef FACERECOGNIZER_H_
#define FACERECOGNIZER_H_

#include "opencv2/contrib/contrib.hpp"
#include <string>

using namespace std;
using namespace cv;

const int EIGEN_METHOD = 0, FISHER_FACES_METHOD = 1, LBPH = 2;

class FaceRecognition {
public:
	FaceRecognition(string& modelName, int recognitionMethod);
	FaceRecognition(string& modelName, string& cascadeFilePath, int recognitionMethod);
	 virtual ~FaceRecognition();

	 bool predictFace(string&, string&, int, double);

	 void createModel(string& modelName, vector<string> sourceImages, vector<int> labelIDs);
	 void updateModel(string& modelName, vector<string> sourceImages, vector<int> labelIDs);


protected:
	 virtual void initRecognitionModel();


private:
	 string cascadeFilePath;
	 string modelPath;
	 int recognitionMethod;
	 Ptr<FaceRecognizer> recognizer;

	 void loadSavedModel();
};

#endif /* FACEDETECTOR_H_ */
