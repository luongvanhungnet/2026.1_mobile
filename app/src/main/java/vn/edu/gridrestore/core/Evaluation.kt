package vn.edu.gridrestore.core

import kotlin.math.max
import kotlin.math.min

/** Coordinates share the original image's canvas; right/bottom are exclusive edges. */
data class Box(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    init {
        require(listOf(left, top, right, bottom).all { it.isFinite() }) { "Box coordinates must be finite" }
        require(right >= left && bottom >= top) { "Box edges are reversed" }
    }

    fun iou(other: Box): Double {
        // Double arithmetic avoids overflow for large but finite Float coordinates.
        val intersection = max(0.0, min(right.toDouble(), other.right.toDouble()) - max(left.toDouble(), other.left.toDouble())) *
            max(0.0, min(bottom.toDouble(), other.bottom.toDouble()) - max(top.toDouble(), other.top.toDouble()))
        val area = (right.toDouble() - left.toDouble()) * (bottom.toDouble() - top.toDouble())
        val otherArea = (other.right.toDouble() - other.left.toDouble()) * (other.bottom.toDouble() - other.top.toDouble())
        val union = area + otherArea - intersection
        return if (union > 0.0) intersection / union else 0.0
    }
}

/** Row-major mask on the original image canvas; true means foreground. */
data class BinaryMask(val width: Int, val height: Int, val pixels: BooleanArray) {
    init {
        require(width > 0 && height > 0) { "Mask dimensions must be positive" }
        require(width.toLong() * height == pixels.size.toLong()) { "Mask pixel count does not match its canvas" }
    }

    fun iou(other: BinaryMask): Double {
        require(width == other.width && height == other.height) { "Masks must use the same original-image canvas" }
        var intersection = 0L
        var union = 0L
        for (index in pixels.indices) {
            if (pixels[index] && other.pixels[index]) intersection++
            if (pixels[index] || other.pixels[index]) union++
        }
        return if (union > 0L) intersection.toDouble() / union else 0.0
    }
}

data class Prediction(
    val imageId: String,
    val classId: Int,
    val confidence: Float,
    val box: Box,
    val mask: BinaryMask? = null
) {
    init {
        require(imageId.isNotBlank()) { "Prediction image ID is missing" }
        require(classId >= 0) { "Class ID must be nonnegative" }
        require(confidence.isFinite() && confidence in 0f..1f) { "Confidence must be finite and between 0 and 1" }
    }
}

data class GroundTruth(
    val imageId: String,
    val classId: Int,
    val box: Box,
    val mask: BinaryMask? = null
) {
    init {
        require(imageId.isNotBlank()) { "Ground-truth image ID is missing" }
        require(classId >= 0) { "Class ID must be nonnegative" }
    }
}

enum class EvaluationTask { DETECTION, SEGMENTATION }

data class ClassMetrics(
    val classId: Int,
    val ap50: Double,
    val ap50To95: Double,
    val groundTruthCount: Int,
    val predictionCount: Int,
    /** Ordered as IoU .50, .55, ..., .95. */
    val apByIou: List<Double>
)

data class EvaluationResult(
    /** Null means there were no ground-truth instances to evaluate. */
    val map50: Double?,
    val map50To95: Double?,
    val perClass: List<ClassMetrics>,
    val predictionCount: Int,
    val groundTruthCount: Int,
    val notes: List<String>
)

/**
 * Dataset-level, class-macro-averaged AP, using 101-point interpolated precision at
 * IoU .50:.05:.95. Confidence sorting is global across images within each class.
 *
 * This implements the COCO-style interpolation and IoU sweep, not COCO's crowd,
 * area ranges, ignore flags or maxDets variants. Inputs must represent one fixed
 * evaluation dataset with the same class mapping for labels and model outputs.
 */
object Evaluator {
    fun evaluate(
        predictions: List<Prediction>,
        groundTruth: List<GroundTruth>,
        task: EvaluationTask = EvaluationTask.DETECTION
    ): EvaluationResult {
        val accumulator = EvaluationAccumulator(task)
        val predictionsByImage = predictions.groupBy { it.imageId }
        val labelsByImage = groundTruth.groupBy { it.imageId }
        (predictionsByImage.keys + labelsByImage.keys).forEach { imageId ->
            accumulator.addImage(imageId, predictionsByImage[imageId].orEmpty(), labelsByImage[imageId].orEmpty())
        }
        return accumulator.result()
    }
}

/**
 * Streams one image at a time, retaining only confidence/TP scalars and label
 * counts. Masks and boxes are never held after addImage returns. Image-local
 * greedy matching is equivalent to matching after global sorting because
 * objects from different images cannot match. Tied confidence follows image
 * insertion order, then prediction input order.
 *
 * Not thread safe: one worker should own each evaluation run.
 */
class EvaluationAccumulator(private val task: EvaluationTask = EvaluationTask.DETECTION) {
    private data class Outcome(val classId: Int, val confidence: Float, val sequence: Long, val matches: BooleanArray)
    private val thresholds = List(10) { 0.5 + it * 0.05 }
    private val seenImages = mutableSetOf<String>()
    private val labelCounts = mutableMapOf<Int, Int>()
    private val outcomes = mutableListOf<Outcome>()
    private var groundTruthCount = 0
    private var predictionCount = 0
    private var nextSequence = 0L

    fun addImage(predictions: List<Prediction>, groundTruth: List<GroundTruth>) {
        val ids = (predictions.map { it.imageId } + groundTruth.map { it.imageId }).distinct()
        if (ids.isEmpty()) return
        require(ids.size == 1) { "addImage accepts exactly one image at a time" }
        addImage(ids.single(), predictions, groundTruth)
    }

    /** Explicit ID also allows recording a negative image with zero predictions. */
    fun addImage(imageId: String, predictions: List<Prediction>, groundTruth: List<GroundTruth>) {
        require(imageId.isNotBlank()) { "Image ID is missing" }
        require(imageId !in seenImages) { "Image '$imageId' has already been evaluated" }
        require(predictions.all { it.imageId == imageId } && groundTruth.all { it.imageId == imageId }) {
            "Predictions and ground truth must belong to image '$imageId'"
        }
        if (task == EvaluationTask.SEGMENTATION) validateMasks(predictions, groundTruth)

        // Stage locally so a validation failure cannot partially add an image.
        val localMatches = Array(predictions.size) { BooleanArray(thresholds.size) }
        val classes = (predictions.map { it.classId } + groundTruth.map { it.classId }).distinct()
        classes.forEach { classId ->
            val labels = groundTruth.filter { it.classId == classId }
            val ranked = predictions.withIndex().filter { it.value.classId == classId }
                .sortedWith(compareByDescending<IndexedValue<Prediction>> { it.value.confidence }.thenBy { it.index })
            // Expensive mask IoU is computed only once per possible object pair.
            val overlaps = ranked.map { indexed ->
                DoubleArray(labels.size) { labelIndex ->
                    when (task) {
                        EvaluationTask.DETECTION -> indexed.value.box.iou(labels[labelIndex].box)
                        EvaluationTask.SEGMENTATION -> indexed.value.mask!!.iou(labels[labelIndex].mask!!)
                    }
                }
            }
            thresholds.forEachIndexed { thresholdIndex, threshold ->
                val matched = BooleanArray(labels.size)
                ranked.forEachIndexed { rank, indexed ->
                    var bestIndex = -1
                    var bestIou = threshold
                    labels.indices.forEach { labelIndex ->
                        val overlap = overlaps[rank][labelIndex]
                        if (!matched[labelIndex] && overlap + 1e-12 >= bestIou) {
                            bestIndex = labelIndex
                            bestIou = overlap
                        }
                    }
                    if (bestIndex >= 0) {
                        matched[bestIndex] = true
                        localMatches[indexed.index][thresholdIndex] = true
                    }
                }
            }
        }

        predictions.forEachIndexed { index, prediction ->
            outcomes += Outcome(prediction.classId, prediction.confidence, nextSequence++, localMatches[index])
        }
        groundTruth.forEach { label -> labelCounts[label.classId] = (labelCounts[label.classId] ?: 0) + 1 }
        predictionCount += predictions.size
        groundTruthCount += groundTruth.size
        seenImages += imageId
    }

    fun result(): EvaluationResult {
        val notes = mutableListOf<String>()
        if (groundTruthCount == 0) {
            return EvaluationResult(null, null, emptyList(), predictionCount, 0,
                listOf("No ground-truth instances; mAP is unavailable."))
        }
        val classes = labelCounts.keys.sorted()
        val absentClassCount = outcomes.count { it.classId !in labelCounts }
        if (absentClassCount > 0) {
            notes += "$absentClassCount prediction(s) belong to classes with no ground truth; those classes are excluded from class-macro mAP."
        }

        val perClass = classes.map { classId ->
            val ranked = outcomes.filter { it.classId == classId }
                .sortedWith(compareByDescending<Outcome> { it.confidence }.thenBy { it.sequence })
            val count = labelCounts.getValue(classId)
            val ap = thresholds.indices.map { thresholdIndex -> averagePrecision(ranked, count, thresholdIndex) }
            ClassMetrics(classId, ap.first(), ap.average(), count, ranked.size, ap)
        }
        notes += "101-point interpolated class-macro AP; IoU .50:.05:.95; no COCO crowd/area/maxDets variants."
        return EvaluationResult(perClass.map { it.ap50 }.average(), perClass.map { it.ap50To95 }.average(),
            perClass, predictionCount, groundTruthCount, notes)
    }

    private fun validateMasks(predictions: List<Prediction>, groundTruth: List<GroundTruth>) {
        require(groundTruth.all { it.mask != null }) { "Segmentation ground truth requires an instance mask for every object" }
        require(predictions.all { it.mask != null }) { "Segmentation predictions require an instance mask for every object" }
        val canvases = mutableMapOf<String, Pair<Int, Int>>()
        fun record(imageId: String, mask: BinaryMask) {
            val canvas = mask.width to mask.height
            val previous = canvases.putIfAbsent(imageId, canvas)
            require(previous == null || previous == canvas) { "All masks for image '$imageId' must use the same original-image canvas" }
        }
        groundTruth.forEach { record(it.imageId, it.mask!!) }
        predictions.forEach { record(it.imageId, it.mask!!) }
    }

    private fun averagePrecision(
        ranked: List<Outcome>,
        labelCount: Int,
        thresholdIndex: Int
    ): Double {
        if (ranked.isEmpty()) return 0.0
        val recall = DoubleArray(ranked.size)
        val precision = DoubleArray(ranked.size)
        var truePositives = 0

        ranked.forEachIndexed { rank, outcome ->
            if (outcome.matches[thresholdIndex]) truePositives++
            recall[rank] = truePositives.toDouble() / labelCount
            precision[rank] = truePositives.toDouble() / (rank + 1)
        }

        // Envelope gives max precision among all detections at or beyond each recall.
        for (index in precision.lastIndex - 1 downTo 0) {
            precision[index] = max(precision[index], precision[index + 1])
        }
        var sum = 0.0
        var rank = 0
        for (point in 0..100) {
            val targetRecall = point / 100.0
            while (rank < recall.size && recall[rank] + 1e-12 < targetRecall) rank++
            if (rank < precision.size) sum += precision[rank]
        }
        return sum / 101.0
    }
}
