/**
 * 
 */
package com.endava.recognitionlibrary;

import org.opencv.contrib.FaceRecognizer;

import android.content.Context;

/**
 * Proxy for JNI face recognition calls to native code.
 * 
 * @author Vasile Chelban
 * 
 */
public class FaceRecognitionProxy {
	private long mNativeObj = 0;

	/**
	 * Create a {@link FaceRecognizer} model for face detection usage, using the provided type of recognition method. <br />
	 * If <code>null</code> is provided for modelType, LBPH recognition method will be used by default.
	 */
	public FaceRecognitionProxy(String modelPath, final RecognitionModel modelType, final boolean createWithFaceDetector, final Context context) {
		if (createWithFaceDetector) {
			final String cascadeFilePath = Utils.initCascadeFile(context);
			if (modelType != null) {
				mNativeObj = createRecognizerWithCascade(modelPath, cascadeFilePath, modelType.ordinal());
			} else {
				mNativeObj = createRecognizerWithCascade(modelPath, cascadeFilePath, RecognitionModel.LBPH.ordinal());
			}
		} else {
			if (modelType != null) {
				mNativeObj = createRecognizer(modelPath, modelType.ordinal());
			} else {
				mNativeObj = createRecognizer(modelPath, RecognitionModel.LBPH.ordinal());
			}
		}
	}

	public void createModel(String modelPath, String[] sourceImages, int[] labelIDs) {
		createModel(mNativeObj, modelPath, sourceImages, labelIDs);
	}

	public void updateModel(String modelPath, String[] sourceImages, int[] labelIDs) {
		updateModel(mNativeObj, modelPath, sourceImages, labelIDs);
	}

	public boolean predictFace(String originalImagePath, String processedImgPath, int contactID, double degree) {
		return predictFace(mNativeObj, originalImagePath, processedImgPath, contactID, degree);
	}

	@Override
	public void finalize() {
		releaseRecognizer(mNativeObj);
	}

	private static native long createRecognizer(String modelPath, final int recognitionModelOrdinal);

	private static native long createRecognizerWithCascade(String modelPath, String cascadeFilePath, final int recognitionModelOrdinal);

	private static native boolean predictFace(long nativeObj, String imagePath, String processedImgPath, int contactID, double degree);

	private static native void createModel(long nativeObj, String modelPath, Object[] sourceImages, int[] labelIDs);

	private static native void updateModel(long nativeObj, String modelPath, Object[] sourceImages, int[] labelIDs);

	// native support for java finalize()
	private static native void releaseRecognizer(long nativeObj);

	public enum RecognitionModel {
		EIGEN_FACES, FISHER_FACES, LBPH
	}
}
