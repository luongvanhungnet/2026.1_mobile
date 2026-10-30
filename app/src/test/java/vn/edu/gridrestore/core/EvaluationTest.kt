package vn.edu.gridrestore.core

import org.junit.Assert.*
import org.junit.Test

class EvaluationTest {
    private val box = Box(0f, 0f, 10f, 10f)

    @Test fun perfectDetectionScoresOneAtEveryThreshold() {
        val result = Evaluator.evaluate(
            listOf(Prediction("a", 0, 0.9f, box)),
            listOf(GroundTruth("a", 0, box))
        )
        assertEquals(1.0, result.map50!!, 1e-9)
        assertEquals(1.0, result.map50To95!!, 1e-9)
    }

    @Test fun confidenceRankingIsGlobalAcrossImagesAndCountsNegativeImageFalsePositives() {
        val result = Evaluator.evaluate(
            listOf(Prediction("negative", 0, 0.99f, box), Prediction("positive", 0, 0.9f, box)),
            listOf(GroundTruth("positive", 0, box))
        )
        assertEquals(0.5, result.map50!!, 1e-9)
    }

    @Test fun oneGroundTruthCannotBeMatchedTwice() {
        val result = Evaluator.evaluate(
            listOf(Prediction("a", 0, 0.99f, box), Prediction("a", 0, 0.98f, box), Prediction("b", 0, 0.97f, box)),
            listOf(GroundTruth("a", 0, box), GroundTruth("b", 0, box))
        )
        assertEquals((51.0 + 50.0 * 2.0 / 3.0) / 101.0, result.map50!!, 1e-9)
    }

    @Test fun unmatchedGroundTruthLimitsMaximumRecall() {
        val result = Evaluator.evaluate(
            listOf(Prediction("a", 0, 1f, box)),
            listOf(GroundTruth("a", 0, box), GroundTruth("b", 0, box))
        )
        assertEquals(51.0 / 101.0, result.map50!!, 1e-9)
    }

    @Test fun metricsSweepIouThresholdsRatherThanRepeatingMap50() {
        val result = Evaluator.evaluate(
            listOf(Prediction("a", 0, 1f, Box(0f, 0f, 6f, 10f))),
            listOf(GroundTruth("a", 0, box))
        )
        assertEquals(1.0, result.map50!!, 1e-9)
        assertEquals(0.3, result.map50To95!!, 1e-9)
        assertEquals(10, result.perClass.single().apByIou.size)
    }

    @Test fun classesUseMacroAverageAndCannotMatchEachOther() {
        val result = Evaluator.evaluate(
            listOf(Prediction("a", 1, 0.99f, box), Prediction("b", 1, 0.9f, box)),
            listOf(GroundTruth("a", 0, box), GroundTruth("b", 1, box))
        )
        // Class 0 has no detections; for class 1, the wrong-image prediction ranks first.
        assertEquals(0.25, result.map50!!, 1e-9)
        assertEquals(listOf(0, 1), result.perClass.map { it.classId })
    }

    @Test fun absentGroundTruthClassesAreExcludedWithAnExplicitNote() {
        val result = Evaluator.evaluate(
            listOf(Prediction("a", 0, 1f, box), Prediction("a", 99, 1f, box)),
            listOf(GroundTruth("a", 0, box))
        )
        assertEquals(1.0, result.map50!!, 1e-9)
        assertEquals(1, result.perClass.size)
        assertTrue(result.notes.any { it.contains("no ground truth") })
    }

    @Test fun noGroundTruthIsUnavailableWhileMissingPredictionsScoresZero() {
        val noGroundTruth = Evaluator.evaluate(listOf(Prediction("a", 0, 1f, box)), emptyList())
        assertNull(noGroundTruth.map50)
        assertNull(noGroundTruth.map50To95)
        assertTrue(noGroundTruth.perClass.isEmpty())
        val noPredictions = Evaluator.evaluate(emptyList(), listOf(GroundTruth("a", 0, box)))
        assertEquals(0.0, noPredictions.map50!!, 1e-9)
    }

    @Test fun segmentationUsesMaskOverlapIndependentlyOfBoundingBoxes() {
        val labelMask = BinaryMask(2, 2, booleanArrayOf(true, true, false, false))
        val wrongMask = BinaryMask(2, 2, booleanArrayOf(false, false, true, true))
        val labels = listOf(GroundTruth("a", 0, box, labelMask))
        val predictions = listOf(Prediction("a", 0, 1f, box, wrongMask))
        assertEquals(1.0, Evaluator.evaluate(predictions, labels).map50!!, 1e-9)
        assertEquals(0.0, Evaluator.evaluate(predictions, labels, EvaluationTask.SEGMENTATION).map50!!, 1e-9)
        val correct = listOf(Prediction("a", 0, 1f, box, labelMask))
        assertEquals(1.0, Evaluator.evaluate(correct, labels, EvaluationTask.SEGMENTATION).map50To95!!, 1e-9)
    }

    @Test fun segmentationRejectsMissingOrMismatchedMasks() {
        val mask = BinaryMask(2, 2, BooleanArray(4) { true })
        expectInvalid {
            Evaluator.evaluate(listOf(Prediction("a", 0, 1f, box)),
                listOf(GroundTruth("a", 0, box, mask)), EvaluationTask.SEGMENTATION)
        }
        expectInvalid {
            Evaluator.evaluate(listOf(Prediction("a", 0, 1f, box, BinaryMask(1, 1, booleanArrayOf(true)))),
                listOf(GroundTruth("a", 0, box, mask)), EvaluationTask.SEGMENTATION)
        }
    }

    @Test fun geometryHandlesDisjointAndEmptyAreasWithoutNan() {
        assertEquals(0.0, box.iou(Box(20f, 20f, 30f, 30f)), 0.0)
        assertEquals(0.0, Box(0f, 0f, 0f, 0f).iou(Box(0f, 0f, 0f, 0f)), 0.0)
        assertEquals(0.0, BinaryMask(1, 1, booleanArrayOf(false)).iou(BinaryMask(1, 1, booleanArrayOf(false))), 0.0)
        assertEquals(1.0, Box(-Float.MAX_VALUE, -Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE)
            .iou(Box(-Float.MAX_VALUE, -Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE)), 1e-9)
    }

    @Test fun invalidInputsFailBeforeProducingMisleadingMetrics() {
        expectInvalid { Box(Float.NaN, 0f, 1f, 1f) }
        expectInvalid { Box(1f, 0f, 0f, 1f) }
        expectInvalid { Prediction("a", 0, Float.NaN, box) }
        expectInvalid { Prediction("a", 0, 1.1f, box) }
        expectInvalid { Prediction("", 0, 1f, box) }
        expectInvalid { GroundTruth("a", -1, box) }
        expectInvalid { BinaryMask(2, 2, booleanArrayOf(true)) }
    }

    @Test fun streamingAccumulatesGlobalRankAndDoesNotRetainMasks() {
        val accumulator = EvaluationAccumulator(EvaluationTask.SEGMENTATION)
        val pixels = booleanArrayOf(true, true, true, true)
        val mask = BinaryMask(2, 2, pixels)
        accumulator.addImage("a", listOf(Prediction("a", 0, 0.7f, box, mask)), listOf(GroundTruth("a", 0, box, mask)))
        pixels.fill(false)
        val negativeMask = BinaryMask(2, 2, booleanArrayOf(true, true, true, true))
        accumulator.addImage("negative", listOf(Prediction("negative", 0, 0.9f, box, negativeMask)), emptyList())
        assertEquals(0.5, accumulator.result().map50!!, 1e-9)
        expectInvalid { accumulator.addImage("a", emptyList(), emptyList()) }
    }

    private fun expectInvalid(block: () -> Unit) {
        try {
            block()
            fail("Expected invalid evaluation input to be rejected")
        } catch (_: IllegalArgumentException) {
            // Expected validation failure.
        }
    }
}
