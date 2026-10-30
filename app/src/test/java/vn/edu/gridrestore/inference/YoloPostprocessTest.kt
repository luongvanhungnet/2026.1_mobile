package vn.edu.gridrestore.inference

import org.junit.Assert.*
import org.junit.Test
import vn.edu.gridrestore.core.Box

class YoloPostprocessTest {
    @Test fun normalizedBoxesUndoPortraitLetterboxAndClipPadding() {
        // A 320x640 source has a 320x640 content rect at x=160 in 640x640.
        val box = YoloPostprocess.originalBox(0.5f, 0.5f, 0.5f, 1f, 640, 640, 320, 640, 160, 0, 1f, 1f)!!
        assertEquals(Box(0f, 0f, 320f, 640f), box)
        assertNull(YoloPostprocess.originalBox(0.05f, 0.5f, 0.05f, 0.1f, 640, 640, 320, 640, 160, 0, 1f, 1f))
    }

    @Test fun nmsPreservesOtherClassesAndHighestConfidence() {
        val box = Box(0f, 0f, 20f, 20f)
        val high = YoloCandidate(1, 0, 0.9f, box)
        val low = YoloCandidate(0, 0, 0.1f, box)
        val otherClass = YoloCandidate(2, 1, 0.8f, box)
        assertEquals(listOf(high, otherClass), YoloPostprocess.nms(listOf(low, otherClass, high)))
    }
}
