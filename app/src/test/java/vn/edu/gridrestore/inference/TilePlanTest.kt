package vn.edu.gridrestore.inference

import org.junit.Assert.*
import org.junit.Test

class TilePlanTest {
    @Test fun coversNonDivisibleEdgesAndSmallImages() {
        for (length in listOf(1, 31, 255, 256, 257, 511, 512, 1001)) {
            val starts = TilePlan.starts(length)
            assertEquals(0, starts.first())
            assertEquals(maxOf(0, length - 256), starts.last())
            for (pixel in 0 until length) assertTrue("uncovered $pixel/$length", starts.any { pixel in it until it + 256 })
            assertEquals(starts.distinct(), starts)
        }
    }
}
