package com.continuum.slm

/**
 * ContinuumEngine — Kotlin JNI bridge to C++ inference engine.
 *
 * Phase D+E: Complete mobile interface.
 * - BPE tokenizer loading (no Python needed)
 * - Streaming token generation via callback
 * - Thermal monitoring
 * - State save/restore for app lifecycle
 * - Thread count control for ARM big.LITTLE
 */
class ContinuumEngine {

    companion object {
        init {
            System.loadLibrary("continuum_jni")
        }

        /**
         * Detect number of big cores on ARM big.LITTLE CPU.
         * Returns recommended thread count (2-4 for most phones).
         */
        external fun detectBigCores(): Int
    }

    /** Load model from CONT binary format file */
    external fun loadModel(path: String): Boolean

    /** Load BPE tokenizer from tokenizer.bin file */
    external fun loadTokenizer(path: String): Boolean

    /** Generate text (non-streaming, returns complete string) */
    external fun generate(prompt: String, maxTokens: Int, temp: Float): String

    /** Generate text with streaming token callback */
    external fun generateStream(
        prompt: String,
        maxTokens: Int,
        temp: Float,
        callback: TokenCallback
    )

    /** Reset conversation state (clear GLT states, window caches, PMB) */
    external fun reset()

    /** Get model info string (dims, layers, vocab, quant type, thread count) */
    external fun getModelInfo(): String

    /** Save conversation state to file (for app lifecycle) */
    external fun saveState(path: String): Boolean

    /** Load conversation state from file */
    external fun loadState(path: String): Boolean

    /** Set number of worker threads (1-4 recommended) */
    external fun setThreadCount(n: Int)

    /** Get thermal status: 0=OK, 1=Warm, 2=Hot, 3=Throttling */
    external fun getThermalStatus(): Int

    /** Check if model is loaded */
    fun isLoaded(): Boolean = _loaded
    private var _loaded = false

    /**
     * Load model and track loaded state.
     * Call this instead of the external loadModel directly.
     */
    fun loadModelTracked(path: String): Boolean {
        _loaded = loadModel(path)
        return _loaded
    }

    /** Token callback for streaming generation */
    interface TokenCallback {
        fun onToken(token: String)
        fun onComplete(text: String)
        fun onError(msg: String)
    }
}
