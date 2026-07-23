package com.continuum.slm

import android.app.Activity
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.method.ScrollingMovementMethod
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import java.io.File

class MainActivity : Activity() {

    private lateinit var chatDisplay: TextView
    private lateinit var inputField: EditText
    private lateinit var sendButton: Button
    private lateinit var statusText: TextView
    private lateinit var resetButton: Button
    private val handler = Handler(Looper.getMainLooper())

    // JNI engine
    private var engine: ContinuumEngine? = null
    private var isGenerating = false

    // ⚡ Phase D: Token callback for streaming
    private val tokenCallback = object : ContinuumEngine.TokenCallback {
        override fun onToken(token: String) {
            handler.post {
                chatDisplay.append(token)
            }
        }
        override fun onComplete(text: String) {
            handler.post {
                chatDisplay.append("\n")
                isGenerating = false
                sendButton.isEnabled = true
                sendButton.text = "Send"
                updateThermalStatus()
            }
        }
        override fun onError(msg: String) {
            handler.post {
                Toast.makeText(this@MainActivity, "Error: $msg", Toast.LENGTH_SHORT).show()
                isGenerating = false
                sendButton.isEnabled = true
                sendButton.text = "Send"
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Build UI programmatically (no XML dependency)
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(16, 16, 16, 16)
        }

        // Status bar
        statusText = TextView(this).apply {
            text = "Loading model..."
            textSize = 12f
            setPadding(8, 8, 8, 8)
        }
        layout.addView(statusText)

        // Chat display (scrollable)
        chatDisplay = TextView(this).apply {
            text = ""
            textSize = 15f
            movementMethod = ScrollingMovementMethod()
            setPadding(12, 12, 12, 12)
            minLines = 12
        }
        val scrollView = ScrollView(this).apply {
            addView(chatDisplay)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                1f
            )
        }
        layout.addView(scrollView)

        // Input field
        inputField = EditText(this).apply {
            hint = "Type your message..."
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }
        layout.addView(inputField)

        // Button row
        val buttonRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }

        sendButton = Button(this).apply {
            text = "Send"
            layoutParams = LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                1f
            )
            setOnClickListener {
                if (!isGenerating) sendMessage()
            }
        }
        buttonRow.addView(sendButton)

        resetButton = Button(this).apply {
            text = "Reset"
            layoutParams = LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                1f
            )
            setOnClickListener {
                engine?.reset()
                chatDisplay.text = ""
                Toast.makeText(this@MainActivity, "Conversation reset", Toast.LENGTH_SHORT).show()
            }
        }
        buttonRow.addView(resetButton)

        layout.addView(buttonRow)
        setContentView(layout)

        // Load model in background
        Thread {
            loadModel()
        }.start()
    }

    private fun loadModel() {
        try {
            System.loadLibrary("continuum_jni")
            engine = ContinuumEngine()

            // Model file paths (copy from assets to internal storage first)
            val modelFile = File(filesDir, "model.bin")
            val tokenizerFile = File(filesDir, "tokenizer.bin")

            // Copy from assets if not present
            if (!modelFile.exists()) {
                assets.open("model.bin").use { input ->
                    modelFile.outputStream().use { output -> input.copyTo(output) }
                }
            }
            if (!tokenizerFile.exists()) {
                assets.open("tokenizer.bin").use { input ->
                    tokenizerFile.outputStream().use { output -> input.copyTo(output) }
                }
            }

            val modelLoaded = engine!!.loadModelTracked(modelFile.absolutePath)
            val tokLoaded = engine!!.loadTokenizer(tokenizerFile.absolutePath)

            // ⚡ Phase B: Auto-detect big cores and set thread count
            val bigCores = ContinuumEngine.detectBigCores()
            engine!!.setThreadCount(bigCores)

            handler.post {
                if (modelLoaded) {
                    val info = engine!!.getModelInfo()
                    statusText.text = "✅ $info | Threads: $bigCores"
                    chatDisplay.text = "Model loaded! Type a message to start chatting.\n\n"
                } else {
                    statusText.text = "❌ Failed to load model"
                }
            }

            // ⚡ Phase E: Restore previous conversation state
            val stateFile = File(filesDir, "conversation_state.bin")
            if (stateFile.exists()) {
                engine!!.loadState(stateFile.absolutePath)
            }

        } catch (e: Exception) {
            handler.post {
                statusText.text = "❌ Error: ${e.message}"
            }
        }
    }

    private fun sendMessage() {
        val prompt = inputField.text.toString().trim()
        if (prompt.isEmpty()) return
        if (engine == null || !engine!!.isLoaded()) {
            Toast.makeText(this, "Model not loaded yet", Toast.LENGTH_SHORT).show()
            return
        }

        isGenerating = true
        sendButton.isEnabled = false
        sendButton.text = "Generating..."

        chatDisplay.append("👤 You: $prompt\n🤖 AI: ")
        inputField.text.clear()

        // ⚡ Phase D: Stream tokens in background thread
        Thread {
            // ⚡ Phase E: Check thermal before starting
            val thermal = engine!!.getThermalStatus()
            if (thermal >= 2) {
                // Hot — reduce thread count temporarily
                engine!!.setThreadCount(1)
            }

            engine!!.generateStream(prompt, 128, 0.8f, tokenCallback)

            // ⚡ Phase E: Save conversation state after each message
            val stateFile = File(filesDir, "conversation_state.bin")
            engine!!.saveState(stateFile.absolutePath)
        }.start()
    }

    private fun updateThermalStatus() {
        val thermal = engine?.getThermalStatus() ?: 0
        val thermalStr = when (thermal) {
            0 -> "OK"
            1 -> "Warm"
            2 -> "Hot"
            3 -> "Throttling"
            else -> "Unknown"
        }
        val info = engine?.getModelInfo() ?: ""
        statusText.text = "$info | Thermal: $thermalStr"
    }

    override fun onPause() {
        super.onPause()
        // ⚡ Phase E: Save state when app goes to background
        if (engine != null && engine!!.isLoaded()) {
            val stateFile = File(filesDir, "conversation_state.bin")
            engine!!.saveState(stateFile.absolutePath)
        }
    }

    override fun onResume() {
        super.onResume()
        // ⚡ Phase E: Restore state when app returns to foreground
        if (engine != null && engine!!.isLoaded()) {
            val stateFile = File(filesDir, "conversation_state.bin")
            if (stateFile.exists()) {
                engine!!.loadState(stateFile.absolutePath)
            }
        }
        updateThermalStatus()
    }
}
