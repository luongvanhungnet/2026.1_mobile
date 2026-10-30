package vn.edu.gridrestore.inference

import kotlin.math.roundToInt

/** Streaming 256-row RGB blend. Call flush(nextTileRow) after finishing a tile row. */
internal class TileBlender(private val width: Int, private val height: Int) {
    val pixels = IntArray(Math.multiplyExact(width, height))
    private val weights = FloatArray(Math.multiplyExact(width, 256))
    private val values = FloatArray(Math.multiplyExact(weights.size, 3))
    private var firstUnflushedRow = 0

    fun add(x: Int, y: Int, sample: (Int) -> Float) {
        require(x >= 0 && x < width && y >= firstUnflushedRow && y < height)
        require(y + minOf(256, height - y) <= firstUnflushedRow + 256) { "Flush earlier rows before adding another tile row" }
        for (dy in 0 until minOf(256, height - y)) {
            for (dx in 0 until minOf(256, width - x)) {
                val weight = TilePlan.blendWeight(dx) * TilePlan.blendWeight(dy)
                val index = ((y + dy) % 256) * width + x + dx
                val outputIndex = (dy * 256 + dx) * 3
                weights[index] += weight
                for (channel in 0..2) {
                    val value = sample(outputIndex + channel)
                    require(value.isFinite()) { "GridFormer produced a non-finite pixel" }
                    values[index * 3 + channel] += value.coerceIn(0f, 1f) * weight
                }
            }
        }
    }

    fun flush(until: Int) {
        require(until in firstUnflushedRow..height)
        while (firstUnflushedRow < until) {
            val globalY = firstUnflushedRow++
            for (x in 0 until width) {
                val index = (globalY % 256) * width + x
                val weight = weights[index]
                check(weight > 0f) { "Internal tile coverage error" }
                val red = (values[index * 3] / weight * 255).roundToInt().coerceIn(0, 255)
                val green = (values[index * 3 + 1] / weight * 255).roundToInt().coerceIn(0, 255)
                val blue = (values[index * 3 + 2] / weight * 255).roundToInt().coerceIn(0, 255)
                pixels[globalY * width + x] = (255 shl 24) or (red shl 16) or (green shl 8) or blue
                weights[index] = 0f
                values[index * 3] = 0f
                values[index * 3 + 1] = 0f
                values[index * 3 + 2] = 0f
            }
        }
    }
}
