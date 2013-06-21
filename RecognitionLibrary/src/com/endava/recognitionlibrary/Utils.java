/**
 * 
 */
package com.endava.recognitionlibrary;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import android.content.Context;
import android.util.Log;

/**
 * @author vchelban
 * 
 */
public final class Utils {

	private static final String LOG_TAG = "Utils";

	public static String initCascadeFile(final Context context) {
		try {
			// load cascade file from application resources
			InputStream is = context.getResources().openRawResource(R.raw.lbpcascade_frontalface);
			File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
			final File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
			if (!mCascadeFile.exists()) {
				FileOutputStream os = new FileOutputStream(mCascadeFile);

				byte[] buffer = new byte[4096];
				int bytesRead;
				while ((bytesRead = is.read(buffer)) != -1) {
					os.write(buffer, 0, bytesRead);
				}
				is.close();
				os.close();
			}
			return mCascadeFile.getAbsolutePath();

		} catch (IOException e) {
			Log.w(LOG_TAG, e.getStackTrace().toString());
		}
		return null;
	}
}
