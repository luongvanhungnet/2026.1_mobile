package vn.edu.gridrestore.inference

import vn.edu.gridrestore.core.Box

internal data class YoloCandidate(val anchor: Int, val classId: Int, val confidence: Float, val box: Box)

internal object YoloPostprocess {
    fun originalBox(
        cx: Float, cy: Float, width: Float, height: Float,
        inputWidth: Int, inputHeight: Int, originalWidth: Int, originalHeight: Int,
        padLeft: Int, padTop: Int, scaleX: Float, scaleY: Float
    ): Box? {
        if (!cx.isFinite() || !cy.isFinite() || !width.isFinite() || !height.isFinite() || width <= 0 || height <= 0) return null
        val centerX = cx * inputWidth
        val centerY = cy * inputHeight
        val boxWidth = width * inputWidth
        val boxHeight = height * inputHeight
        val x1 = ((centerX - boxWidth / 2 - padLeft) / scaleX).coerceIn(0f, originalWidth.toFloat())
        val y1 = ((centerY - boxHeight / 2 - padTop) / scaleY).coerceIn(0f, originalHeight.toFloat())
        val x2 = ((centerX + boxWidth / 2 - padLeft) / scaleX).coerceIn(0f, originalWidth.toFloat())
        val y2 = ((centerY + boxHeight / 2 - padTop) / scaleY).coerceIn(0f, originalHeight.toFloat())
        return if (x2 > x1 && y2 > y1) Box(x1, y1, x2, y2) else null
    }

    fun nms(candidates: List<YoloCandidate>): List<YoloCandidate> {
        val selected = mutableListOf<YoloCandidate>()
        for (candidate in candidates.sortedByDescending { it.confidence }.take(30000)) {
            if (selected.none { it.classId == candidate.classId && it.box.iou(candidate.box) > 0.7 }) {
                selected += candidate
                if (selected.size == 100) break
            }
        }
        return selected
    }
}
