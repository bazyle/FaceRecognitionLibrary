/**
 * 
 */
package com.endava.recognitionlibrary;

import org.opencv.core.MatOfRect;

import android.content.Context;

/**
 * Proxy for JNI face detection calls to native code.
 * 
 * @author Vasile Chelban
 * 
 */
public final class FaceProcessingProxy {

	/**
	 * private c-tor to avoid inheritance
	 */
	private FaceProcessingProxy() {
	}

	public static MatOfRect detectFaces(String inputImagePath, int faceSize, final Context context) {
		MatOfRect faces = new MatOfRect();
		final String cascadeName = Utils.initCascadeFile(context);
		processImageAndDetectFaces(inputImagePath, cascadeName, faces.getNativeObjAddr(), faceSize);
		return faces;
	}

	public static boolean cropImage(final String originalImagePath, final String croppedImagePath, int imgX, int imgY, int faceWidth, int faceHeight) {
		return nativeCropFace(originalImagePath, croppedImagePath, imgX, imgY, faceWidth, faceHeight);
	}

	private static native boolean nativeCropFace(String originalImagePath, String imageFilePath, int imgX, int imgY, int faceWidth, int faceHeight);

	private static native void processImageAndDetectFaces(String inputImagePath, String cascadeName, long imageDataAddr, int faceSize);
}
