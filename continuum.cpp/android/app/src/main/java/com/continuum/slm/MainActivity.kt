package com.continuum.slm

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*

/**
 * MainActivity — Simple chat UI for Continuum SLM on Android.
 *
 * Demonstrates JNI integration:
 *   1. Load the .so library
 *   2. Load the GGUF model
 *   3. Generate text responses
 *
 * Setup:
 *   1. Copy libcontinuum_jni.so to app/src/main/jniLibs/arm64-v8a/
 *   2. Copy model.gguf to app/src/main/assets/
 *   3. Add this class to your Android project
 */
class MainActivity : AppCompatActivity() {

    private lateinit var inputEditText: EditText
    private lateinit var sendButton: Button
    private lateinit var outputTextView: TextView
    private lateinit var statusTextView: TextView

    private var engineLoaded = false
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputEditText = findViewById(R.id.inputText)
        sendButton = findViewById(R.id.sendButton)
        outputTextView = findViewById(R.id.outputText)
        statusTextView = findViewById(R.id.statusText)

        // Load model from assets
        loadModel()

        sendButton.setOnClickListener {
            val prompt = inputEditText.text.toString()
            if (prompt.isNotBlank() && engineLoaded) {
                generate(prompt)
            }
        }
    }

    private fun loadModel() {
        scope.launch {
            try {
                statusTextView.text = "Loading model..."
                // Copy from assets to internal storage (GGUF can be large)
                val modelPath = copyAssetToInternal("model.gguf")

                val ok = ContinuumEngine.loadModel(modelPath)
                if (ok) {
                    engineLoaded = true
                    withContext(Dispatchers.Main) {
                        statusTextView.text = ContinuumEngine.getModelInfo()
                        sendButton.isEnabled = true
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        statusTextView.text = "ERROR: Failed to load model"
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    statusTextView.text = "ERROR: ${e.message}"
                }
            }
        }
    }

    private fun generate(prompt: String) {
        scope.launch {
            try {
                withContext(Dispatchers.Main) {
                    statusTextView.text = "Generating..."
                    sendButton.isEnabled = false
                }

                val response = ContinuumEngine.generate(
                    prompt = prompt,
                    maxTokens = 100,
                    temperature = 0.8f
                )

                withContext(Dispatchers.Main) {
                    outputTextView.text = response
                    statusTextView.text = "Done"
                    sendButton.isEnabled = true
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    statusTextView.text = "ERROR: ${e.message}"
                    sendButton.isEnabled = true
                }
            }
        }
    }

    private fun copyAssetToInternal(filename: String): String {
        val outFile = java.io.File(filesDir, filename)
        if (outFile.exists()) return outFile.absolutePath

        assets.open(filename).use { input ->
            outFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        return outFile.absolutePath
    }

    override fun onDestroy() {
        scope.cancel()
        ContinuumEngine.reset()
        super.onDestroy()
    }
}

// ============================================================================
// JNI Bridge — matches continuum_jni.cpp exports
// ============================================================================
object ContinuumEngine {
    init {
        System.loadLibrary("continuum_jni")
    }

    external fun loadModel(path: String): Boolean
    external fun generate(prompt: String, maxTokens: Int, temperature: Float): String
    external fun reset()
    external fun getModelInfo(): String
}
