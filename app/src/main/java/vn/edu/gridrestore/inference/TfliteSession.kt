package vn.edu.gridrestore.inference

import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import org.json.JSONObject

enum class Backend { CPU, GPU }

data class BenchmarkResult(
    val backend: Backend,
    val warmups: Int,
    val runs: Int,
    val meanMs: Double,
    val minMs: Double,
    val maxMs: Double,
    val stdDevMs: Double,
    val delegation: String,
    val samplesMs: List<Double> = emptyList(),
    val placement: JSONObject? = null,
    val inputDescription: String = "256×256 RGB FP32; inference only, excludes decode, tiling and display"
)

/** GPU delegate creation, invocation and destruction must happen on one thread. */
internal class TfliteSession(model: File, val backend: Backend, threads: Int) : Closeable {
    private val ownerThread = Thread.currentThread().id
    private var delegate: GpuDelegate? = null
    private var before: DelegateProbe? = null
    private var after: DelegateProbe? = null
    val interpreter: Interpreter

    init {
        require(model.isFile && model.length() > 0) { "Missing model: ${model.name}. Import the FP32 TFLite model first." }
        require(threads in 1..16)
        var candidate: Interpreter? = null
        try {
            val options = Interpreter.Options().setNumThreads(threads).setUseXNNPACK(backend == Backend.CPU)
            if (backend == Backend.GPU) {
                delegate = GpuDelegate(GpuDelegate.Options().setPrecisionLossAllowed(false))
                before = DelegateProbe(delegate!!.nativeHandle)
                after = DelegateProbe(delegate!!.nativeHandle)
                options.setUseXNNPACK(false)
                options.addDelegate(before!!)
                options.addDelegate(delegate!!)
                options.addDelegate(after!!)
            }
            val mapped = FileInputStream(model).use { input ->
                input.channel.map(FileChannel.MapMode.READ_ONLY, 0, model.length())
            }
            candidate = Interpreter(mapped, options)
            interpreter = candidate
            interpreter.allocateTensors()
        } catch (error: Throwable) {
            candidate?.close()
            delegate?.close()
            before?.close(); after?.close()
            throw IllegalArgumentException("Cannot load ${model.name} with $backend: ${error.message}", error)
        }
    }

    fun checkThread() = check(Thread.currentThread().id == ownerThread) { "Create, run and close the interpreter on the same worker thread" }

    fun placement(): JSONObject = if (backend == Backend.CPU) JSONObject().put("classification", "CPU_ONLY").put("gpuPartitions", 0)
        else JSONObject().put("before", before!!.evidence()).put("after", after!!.evidence()).also { result ->
            val observed = result.getJSONObject("after")
            val original = result.getJSONObject("before")
            val kind = when {
                !original.getBoolean("prepared") || !observed.getBoolean("prepared") -> "UNVERIFIED"
                original.getInt("executionNodes") != observed.getInt("gpuOriginalNodes") + observed.getInt("cpuNodes") -> "UNVERIFIED"
                observed.getInt("gpuPartitions") == 0 -> "CPU_ONLY"
                observed.getInt("cpuNodes") == 0 -> "GPU_FULL"
                else -> "GPU_MIXED"
            }
            result.put("classification", kind).put("source", "TfLiteContext execution plan after GPU ModifyGraphWithDelegate; XNNPACK disabled for GPU request")
        }

    fun requireFloatTensor(input: Boolean, index: Int) {
        val tensor = if (input) interpreter.getInputTensor(index) else interpreter.getOutputTensor(index)
        require(tensor.dataType() == DataType.FLOAT32) { "FP32 model required: ${tensor.name()} is ${tensor.dataType()}" }
    }

    override fun close() {
        checkThread()
        try { interpreter.close() } finally { delegate?.close(); before?.close(); after?.close() }
    }
}

internal fun floatBuffer(elements: Int): ByteBuffer = ByteBuffer.allocateDirect(Math.multiplyExact(elements, 4)).order(ByteOrder.nativeOrder())
