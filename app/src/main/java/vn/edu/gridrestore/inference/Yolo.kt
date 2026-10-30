package vn.edu.gridrestore.inference

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import vn.edu.gridrestore.core.BinaryMask
import vn.edu.gridrestore.core.EvaluationTask
import vn.edu.gridrestore.core.Prediction
import java.io.Closeable
import java.io.File
import kotlin.math.floor
import kotlin.math.roundToInt

/** Ultralytics YOLOv8/YOLO11 FP32 TFLite, nms=false, normalized xywh outputs. */
class Yolo(
    model: File,
    private val task: EvaluationTask,
    backend: Backend = Backend.CPU,
    threads: Int = 4
) : Closeable {
    private val session = TfliteSession(model, backend, threads)
    private val interpreter get() = session.interpreter
    private val inputShape: IntArray
    private val detectionIndex: Int
    private val prototypeIndex: Int?
    private val channels: Int
    private val anchors: Int
    private val maskChannels: Int
    private val prototypeHeight: Int
    private val prototypeWidth: Int
    private val prototypeNhwc: Boolean
    private val input: java.nio.ByteBuffer
    private val outputs: Map<Int, java.nio.ByteBuffer>
    val classCount: Int

    init {
        try {
            require(interpreter.inputTensorCount == 1) { "YOLO must have one input tensor" }
            session.requireFloatTensor(true, 0)
            inputShape = interpreter.getInputTensor(0).shape()
            require(inputShape.contentEquals(intArrayOf(1, 640, 640, 3))) {
                "YOLO input must be [1,640,640,3] RGB FP32"
            }
            val detectionIndices = (0 until interpreter.outputTensorCount).filter { interpreter.getOutputTensor(it).shape().size == 3 }
            require(detectionIndices.size == 1) { "Expected one YOLO raw [1,4+classes(+masks),anchors] output; export with nms=false" }
            detectionIndex = detectionIndices.single()
            val detectionShape = interpreter.getOutputTensor(detectionIndex).shape()
            require(detectionShape[0] == 1 && detectionShape[1] >= 5 && detectionShape[1] < detectionShape[2]) { "Unsupported YOLO detection tensor layout" }
            channels = detectionShape[1]
            anchors = detectionShape[2]
            prototypeIndex = if (task == EvaluationTask.SEGMENTATION) {
                val indices = (0 until interpreter.outputTensorCount).filter { interpreter.getOutputTensor(it).shape().size == 4 }
                require(indices.size == 1 && interpreter.outputTensorCount == 2) { "Segmentation requires one raw detection output and one mask prototype output" }
                indices.single()
            } else {
                require(interpreter.outputTensorCount == 1) { "Select a detection model for the detection task" }
                null
            }
            if (prototypeIndex != null) {
                val shape = interpreter.getOutputTensor(prototypeIndex).shape()
                require(shape[0] == 1)
                prototypeNhwc = shape[3] <= 128 && shape[3] < shape[1] && shape[3] < shape[2]
                if (prototypeNhwc) {
                    prototypeHeight = shape[1]; prototypeWidth = shape[2]; maskChannels = shape[3]
                } else {
                    require(shape[1] <= 128 && shape[1] < shape[2] && shape[1] < shape[3]) { "Unsupported mask prototype layout" }
                    maskChannels = shape[1]; prototypeHeight = shape[2]; prototypeWidth = shape[3]
                }
            } else {
                prototypeNhwc = true; prototypeHeight = 0; prototypeWidth = 0; maskChannels = 0
            }
            classCount = channels - 4 - maskChannels
            require(classCount == 80 && anchors == 8400 && (prototypeIndex == null || maskChannels == 32)) {
                "Expected COCO-80 YOLOv8/11 raw TFLite ABI (8400 anchors, 32 mask coefficients for segmentation)"
            }
            input = floatBuffer(interpreter.getInputTensor(0).numElements())
            outputs = (0 until interpreter.outputTensorCount).associateWith {
                session.requireFloatTensor(false, it)
                floatBuffer(interpreter.getOutputTensor(it).numElements())
            }
        } catch (error: Throwable) {
            session.close()
            throw error
        }
    }

    fun predict(source: Bitmap, imageId: String): List<Prediction> {
        session.checkThread()
        require(imageId.isNotBlank())
        require(!source.isRecycled)
        val inputHeight = inputShape[1]
        val inputWidth = inputShape[2]
        val ratio = minOf(inputWidth.toFloat() / source.width, inputHeight.toFloat() / source.height)
        val scaledWidth = (source.width * ratio).roundToInt().coerceIn(1, inputWidth)
        val scaledHeight = (source.height * ratio).roundToInt().coerceIn(1, inputHeight)
        val left = (inputWidth - scaledWidth) / 2
        val top = (inputHeight - scaledHeight) / 2
        val scaleX = scaledWidth.toFloat() / source.width
        val scaleY = scaledHeight.toFloat() / source.height
        val letterbox = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        try {
            Canvas(letterbox).apply {
                drawColor(Color.rgb(114, 114, 114))
                drawBitmap(source, null, RectF(left.toFloat(), top.toFloat(), (left + scaledWidth).toFloat(), (top + scaledHeight).toFloat()), Paint(Paint.FILTER_BITMAP_FLAG))
            }
            val row = IntArray(inputWidth)
            input.clear()
            for (y in 0 until inputHeight) {
                letterbox.getPixels(row, 0, inputWidth, 0, y, inputWidth, 1)
                for (color in row) {
                    input.putFloat(((color ushr 16) and 255) / 255f)
                    input.putFloat(((color ushr 8) and 255) / 255f)
                    input.putFloat((color and 255) / 255f)
                }
            }
        } finally { letterbox.recycle() }
        input.rewind()
        outputs.values.forEach { it.clear() }
        interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs.mapValues { it.value as Any })
        val detection = outputs.getValue(detectionIndex)
        fun value(channel: Int, anchor: Int): Float = detection.getFloat((channel * anchors + anchor) * 4)
        val candidates = mutableListOf<YoloCandidate>()
        for (anchor in 0 until anchors) {
            if (Thread.currentThread().isInterrupted) throw InterruptedException("YOLO evaluation cancelled")
            var bestClass = 0
            var bestScore = value(4, anchor)
            for (classId in 1 until classCount) {
                val score = value(4 + classId, anchor)
                if (score > bestScore) { bestClass = classId; bestScore = score }
            }
            if (!bestScore.isFinite() || bestScore < 0.001f || bestScore > 1f) continue
            val box = YoloPostprocess.originalBox(value(0, anchor), value(1, anchor), value(2, anchor), value(3, anchor),
                inputWidth, inputHeight, source.width, source.height, left, top, scaleX, scaleY)
            if (box != null) candidates += YoloCandidate(anchor, bestClass, bestScore, box)
        }
        // Pre-NMS cap limits worst-case quadratic work on corrupted or untrained models.
        val selected = YoloPostprocess.nms(candidates)
        return selected.map { candidate ->
            val mask = if (prototypeIndex == null) null else {
                val coefficients = FloatArray(maskChannels) { value(4 + classCount + it, candidate.anchor) }
                require(coefficients.all { it.isFinite() }) { "YOLO produced non-finite mask coefficients" }
                val prototype = outputs.getValue(prototypeIndex)
                val maskLogits = FloatArray(prototypeWidth * prototypeHeight)
                for (y in 0 until prototypeHeight) for (x in 0 until prototypeWidth) {
                    var logit = 0f
                    for (channel in 0 until maskChannels) {
                        val index = if (prototypeNhwc) (y * prototypeWidth + x) * maskChannels + channel
                        else (channel * prototypeHeight + y) * prototypeWidth + x
                        logit += coefficients[channel] * prototype.getFloat(index * 4)
                    }
                    require(logit.isFinite()) { "YOLO produced non-finite mask logits" }
                    maskLogits[y * prototypeWidth + x] = logit
                }
                val pixels = BooleanArray(Math.multiplyExact(source.width, source.height))
                val box = candidate.box
                val minX = floor(box.left).toInt().coerceAtLeast(0)
                val minY = floor(box.top).toInt().coerceAtLeast(0)
                val maxX = kotlin.math.ceil(box.right).toInt().coerceAtMost(source.width)
                val maxY = kotlin.math.ceil(box.bottom).toInt().coerceAtMost(source.height)
                for (y in minY until maxY) {
                    if (Thread.currentThread().isInterrupted) throw InterruptedException("Mask decoding cancelled")
                    for (x in minX until maxX) {
                        val px = ((left + (x + 0.5f) * scaleX) * prototypeWidth / inputWidth - 0.5f).coerceIn(0f, (prototypeWidth - 1).toFloat())
                        val py = ((top + (y + 0.5f) * scaleY) * prototypeHeight / inputHeight - 0.5f).coerceIn(0f, (prototypeHeight - 1).toFloat())
                        val x0 = floor(px).toInt(); val y0 = floor(py).toInt()
                        val x1 = minOf(x0 + 1, prototypeWidth - 1); val y1 = minOf(y0 + 1, prototypeHeight - 1)
                        val dx = px - x0; val dy = py - y0
                        val score = maskLogits[y0 * prototypeWidth + x0] * (1 - dx) * (1 - dy) +
                            maskLogits[y0 * prototypeWidth + x1] * dx * (1 - dy) +
                            maskLogits[y1 * prototypeWidth + x0] * (1 - dx) * dy +
                            maskLogits[y1 * prototypeWidth + x1] * dx * dy
                        pixels[y * source.width + x] = score > 0f
                    }
                }
                BinaryMask(source.width, source.height, pixels)
            }
            Prediction(imageId, candidate.classId, candidate.confidence, candidate.box, mask)
        }
    }

    override fun close() = session.close()
}
