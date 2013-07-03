/**
 * 
 */
package com.endava.recognitionlibrary.proxy;

import org.opencv.core.MatOfRect;

import com.endava.recognitionlibrary.Utils;

import android.content.Context;


/**
 * Proxy for JNI face detection and image processing calls to native code.
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

	/**
	 * Checks the Exif information of the specified image and fixes the rotation degree, if needed.
	 * 
	 * @param originalImgPath
	 *            - the path of the image to be rotated, if needed.
	 * @param rotationDegree
	 *            - the degree image is rotated with. The image will be rotated, actually with (360 - rotationDegree)
	 *            degrees.
	 * 
	 * @return - <code>true</code> - if the image was rotated and saved, <code>false</code> - otherwise.
	 */
	public static void fixImageRotation(final String originalImgPath, double rotationDegree) {
		rotateImage(originalImgPath, rotationDegree);
	}

	private static native boolean nativeCropFace(String originalImagePath, String imageFilePath, int imgX, int imgY, int faceWidth, int faceHeight);

	private static native void processImageAndDetectFaces(String inputImagePath, String cascadeName, long imageDataAddr, int faceSize);

	private static native boolean rotateImage(String inputImagePath, double rotationDegree);
}
