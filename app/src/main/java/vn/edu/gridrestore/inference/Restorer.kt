package vn.edu.gridrestore.inference

import android.graphics.Bitmap
import java.io.Closeable
import java.io.File
import kotlin.math.sqrt

class Restorer(model: File, backend: Backend = Backend.CPU, threads: Int = 4) : Closeable {
    private val session = TfliteSession(model, backend, threads)
    private val input = floatBuffer(256 * 256 * 3)
    private val output = floatBuffer(256 * 256 * 3)
    private val row = IntArray(256)

    init {
        try {
            val interpreter = session.interpreter
            require(interpreter.inputTensorCount == 1 && interpreter.outputTensorCount == 1) { "GridFormer must have one input and one output" }
            session.requireFloatTensor(true, 0)
            session.requireFloatTensor(false, 0)
            val expected = intArrayOf(1, 256, 256, 3)
            require(interpreter.getInputTensor(0).shape().contentEquals(expected)) { "GridFormer input must be [1,256,256,3] RGB NHWC FP32" }
            require(interpreter.getOutputTensor(0).shape().contentEquals(expected)) { "GridFormer output must be [1,256,256,3] RGB NHWC FP32" }
        } catch (error: Throwable) {
            session.close()
            throw error
        }
    }

    private fun loadTile(source: Bitmap, x: Int, y: Int) {
        input.clear()
        val actualWidth = minOf(256, source.width - x)
        for (dy in 0 until 256) {
            source.getPixels(row, 0, actualWidth, x, minOf(y + dy, source.height - 1), actualWidth, 1)
            for (dx in 0 until 256) {
                val color = row[minOf(dx, actualWidth - 1)]
                input.putFloat(((color ushr 16) and 255) / 255f)
                input.putFloat(((color ushr 8) and 255) / 255f)
                input.putFloat((color and 255) / 255f)
            }
        }
        input.rewind()
    }

    private fun runTile() {
        input.rewind()
        output.clear()
        session.interpreter.run(input, output)
        output.rewind()
    }

    /** Uses a 256-row accumulation ring, rather than float buffers for the whole image. */
    fun restore(source: Bitmap, onProgress: (Int, Int) -> Unit = { _, _ -> }): Bitmap {
        session.checkThread()
        require(!source.isRecycled)
        val width = source.width
        val height = source.height
        val xs = TilePlan.starts(width)
        val ys = TilePlan.starts(height)
        val blender = TileBlender(width, height)
        var done = 0
        for ((yi, y) in ys.withIndex()) {
            for (x in xs) {
                if (Thread.currentThread().isInterrupted) throw InterruptedException("Restoration cancelled")
                loadTile(source, x, y)
                runTile()
                blender.add(x, y) { output.getFloat(it * 4) }
                onProgress(++done, xs.size * ys.size)
            }
            blender.flush(if (yi < ys.lastIndex) ys[yi + 1] else height)
        }
        return Bitmap.createBitmap(blender.pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    fun benchmark(source: Bitmap, onProgress: (Int, Int) -> Unit = { _, _ -> }): BenchmarkResult {
        session.checkThread()
        loadTile(source, maxOf(0, (source.width - 256) / 2), maxOf(0, (source.height - 256) / 2))
        repeat(5) {
            if (Thread.currentThread().isInterrupted) throw InterruptedException("Benchmark cancelled")
            runTile()
            if (it == 0) {
                for (index in 0 until 256 * 256 * 3) require(output.getFloat(index * 4).isFinite()) {
                    "GridFormer produced non-finite pixels during benchmark warm-up"
                }
            }
            onProgress(it + 1, 55)
        }
        val samples = DoubleArray(50) {
            if (Thread.currentThread().isInterrupted) throw InterruptedException("Benchmark cancelled")
            val start = System.nanoTime()
            runTile()
            val elapsed = (System.nanoTime() - start) / 1_000_000.0
            onProgress(it + 6, 55)
            elapsed
        }
        val average = samples.average()
        return BenchmarkResult(session.backend, 5, 50, average, samples.min(), samples.max(),
            sqrt(samples.sumOf { (it - average) * (it - average) } / samples.size),
            if (session.backend == Backend.CPU) "CPU requested; no GPU delegate attached"
            else session.placement().getString("classification"), samples.toList(), session.placement())
    }

    override fun close() = session.close()
}
