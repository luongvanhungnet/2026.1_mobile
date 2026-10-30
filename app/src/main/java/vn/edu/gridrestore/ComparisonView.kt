package vn.edu.gridrestore

import android.content.Context
import android.graphics.*
import android.view.MotionEvent
import android.view.View

class ComparisonView(context: Context) : View(context) {
    var before: Bitmap? = null
    var after: Bitmap? = null
    private var fraction = .5f
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
    private val density = resources.displayMetrics.density
    private val rect = RectF()
    private val destination = RectF()
    private val clip = Path()
    init { contentDescription = "So sánh ảnh trước và sau khôi phục. Kéo ngang để thay đổi đường chia."; isFocusable = true }
    override fun onDraw(canvas: Canvas) {
        rect.set(0f, 0f, width.toFloat(), height.toFloat())
        canvas.save()
        clip.rewind(); clip.addRoundRect(rect, 20 * density, 20 * density, Path.Direction.CW)
        canvas.clipPath(clip)
        canvas.drawColor(Color.rgb(27, 38, 48))
        val source = before
        if (source == null) {
            paint.color = Color.rgb(46, 65, 75); paint.strokeWidth = density
            for (x in 0..width step (32 * density).toInt().coerceAtLeast(1)) canvas.drawLine(x.toFloat(), 0f, x.toFloat(), height.toFloat(), paint)
            for (y in 0..height step (32 * density).toInt().coerceAtLeast(1)) canvas.drawLine(0f, y.toFloat(), width.toFloat(), y.toFloat(), paint)
            paint.color = Color.rgb(128, 229, 188); paint.textSize = 32 * density; paint.textAlign = Paint.Align.CENTER
            canvas.drawText("▧", width / 2f, height / 2f - 8 * density, paint)
            paint.textSize = 14 * density; paint.color = Color.rgb(159, 178, 188)
            canvas.drawText("Ảnh của bạn, rõ nét hơn", width / 2f, height / 2f + 26 * density, paint)
        } else {
            val scale = minOf(width.toFloat() / source.width, height.toFloat() / source.height)
            val w = source.width * scale; val h = source.height * scale
            destination.set((width - w) / 2, (height - h) / 2, (width + w) / 2, (height + h) / 2)
            val dst = destination
            canvas.drawBitmap(source, null, dst, paint)
            after?.let {
                canvas.save(); canvas.clipRect(width * fraction, 0f, width.toFloat(), height.toFloat())
                canvas.drawBitmap(it, null, dst, paint); canvas.restore()
                paint.color = Color.rgb(128, 229, 188); paint.strokeWidth = 2 * density
                canvas.drawLine(width * fraction, 0f, width * fraction, height.toFloat(), paint)
                canvas.drawCircle(width * fraction, height / 2f, 16 * density, paint)
                paint.color = Color.rgb(16, 24, 32); paint.textAlign = Paint.Align.CENTER; paint.textSize = 14 * density
                canvas.drawText("↔", width * fraction, height / 2f + 5 * density, paint)
            }
            paint.color = Color.WHITE; paint.textSize = 11 * density; paint.textAlign = Paint.Align.LEFT
            canvas.drawText("GỐC", 14 * density, 24 * density, paint)
            if (after != null) { paint.textAlign = Paint.Align.RIGHT; canvas.drawText("KHÔI PHỤC", width - 14 * density, 24 * density, paint) }
        }
        canvas.restore()
    }
    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (after == null) return super.onTouchEvent(event)
        when (event.action) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> { parent.requestDisallowInterceptTouchEvent(true); fraction = (event.x / width).coerceIn(.05f, .95f); invalidate(); return true }
            MotionEvent.ACTION_UP -> { performClick(); parent.requestDisallowInterceptTouchEvent(false); return true }
        }
        return true
    }
    override fun performClick(): Boolean { super.performClick(); return true }
}
