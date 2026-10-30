package vn.edu.gridrestore.inference

/** Tile origins cover every pixel; the last tile is anchored to the far edge. */
object TilePlan {
    const val SIZE = 256
    const val OVERLAP = 32

    fun starts(length: Int): List<Int> {
        require(length > 0) { "Image dimensions must be positive" }
        if (length <= SIZE) return listOf(0)
        val result = mutableListOf(0)
        while (result.last() + SIZE < length) {
            result += minOf(result.last() + SIZE - OVERLAP, length - SIZE)
        }
        return result
    }

    fun blendWeight(offset: Int): Float = minOf(offset + 1, SIZE - offset, OVERLAP).toFloat() / OVERLAP
}
