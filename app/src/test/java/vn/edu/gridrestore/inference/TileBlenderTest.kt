package vn.edu.gridrestore.inference

import org.junit.Assert.*
import org.junit.Test

class TileBlenderTest {
    @Test fun identityTilesReassembleExactPixelsIncludingUnevenOverlapsAndEdges() {
        for ((width, height) in listOf(1 to 1, 31 to 19, 257 to 259, 513 to 481)) {
            fun color(x: Int, y: Int): Int = (255 shl 24) or ((x % 256) shl 16) or ((y % 256) shl 8) or ((x + y) % 256)
            val blender = TileBlender(width, height)
            val ys = TilePlan.starts(height)
            for ((yi, y) in ys.withIndex()) {
                for (x in TilePlan.starts(width)) {
                    blender.add(x, y) { index ->
                        val pixel = index / 3
                        val channel = index % 3
                        val actualX = minOf(x + pixel % 256, width - 1)
                        val actualY = minOf(y + pixel / 256, height - 1)
                        ((color(actualX, actualY) ushr ((2 - channel) * 8)) and 255) / 255f
                    }
                }
                blender.flush(if (yi == ys.lastIndex) height else ys[yi + 1])
            }
            val expected = IntArray(width * height) { color(it % width, it / width) }
            assertArrayEquals("$width x $height", expected, blender.pixels)
        }
    }
}
