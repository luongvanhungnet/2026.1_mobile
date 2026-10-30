package vn.edu.gridrestore

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assert.assertEquals
import android.graphics.Bitmap
import android.os.Bundle
import org.json.JSONObject
import org.json.JSONArray
import vn.edu.gridrestore.data.*
import vn.edu.gridrestore.inference.*
import java.io.File

/** Real models/device only. Results persist in the target app for UI/export. */
@Suppress("DEPRECATION")
@RunWith(AndroidJUnit4::class)
class DeviceWorkflowTest {
    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()
    private fun update(message: String) {
        val context = instrumentation.targetContext
        File(context.filesDir, "device-progress.txt").writeText(message)
        instrumentation.sendStatus(1, Bundle().apply { putString("stream", "$message\n") })
    }
    @Test fun testRealModelWorkflow() {
        val context = instrumentation.targetContext
        val files = context.filesDir
        try {
            BundledModels.installMissing(context, ::update)
            val input = File("/data/local/tmp/gridrestore-test.zip")
            require(input.isFile) { "Push the labelled dataset ZIP to /data/local/tmp/gridrestore-test.zip first" }
            val dataset = input.inputStream().use { DatasetImporter.importZip(it, files) }
            val source = dataset.samples.first().bitmap()
            try {
                File(files, "input.png").outputStream().use { source.compress(Bitmap.CompressFormat.PNG, 100, it) }
                update("CPU restoration smoke test")
                val restored = Restorer(File(files, "models/gridformer.tflite")).use { it.restore(source) }
                assertEquals(source.width, restored.width); assertEquals(source.height, restored.height)
                File(files, "restored.png").outputStream().use { restored.compress(Bitmap.CompressFormat.PNG, 100, it) }
                restored.recycle()
                update("Real model edge sizes / multi-tile stitching")
                val edgeSizes=JSONArray()
                Restorer(File(files,"models/gridformer.tflite")).use { runner ->
                    listOf(31 to 19, 257 to 259).forEach { (width,height) ->
                        val edgeSource=Bitmap.createScaledBitmap(source,width,height,true)
                        try {
                            val output=runner.restore(edgeSource)
                            try { assertEquals(width,output.width);assertEquals(height,output.height) }
                            finally { output.recycle() }
                        } finally { edgeSource.recycle() }
                        edgeSizes.put(JSONObject().put("width",width).put("height",height).put("status","passed"))
                    }
                }
                File(files,"edge-size-validation.json").writeText(JSONObject().put("sizes",edgeSizes).toString(2))
                update("Dataset evaluation")
                val evaluation = Experiments.evaluate(files, dataset, ::update)
                File(files, "evaluation.json").writeText(evaluation.toString(2))
                update("CPU/GPU 5 warmups + 50 measured runs")
                val runBenchmark = InstrumentationRegistry.getArguments().getString("runBenchmark") == "true"
                if (runBenchmark) {
                    val benchmark = Experiments.benchmark(files, source, ::update, context)
                    File(files, "benchmark.json").writeText(benchmark.toString(2))
                    assertEquals("measured", benchmark.getJSONArray("cases").getJSONObject(0).getString("status"))
                    assertEquals(50, benchmark.getJSONArray("cases").getJSONObject(0).getJSONArray("samplesMs").length())
                }
                File(files, "device-validation.json").writeText(JSONObject().put("status", "passed").put("device", Experiments.device()).put("benchmarkExecuted",runBenchmark).toString(2))
                update("DONE: Android model workflow completed; benchmarkExecuted=$runBenchmark")
            } finally { source.recycle() }
        } catch (error: Throwable) {
            File(files, "device-validation.json").writeText(JSONObject().put("status", "failed").put("error", error.toString()).put("stack", error.stackTraceToString()).toString(2))
            update("FAILED: $error")
            throw error
        }
    }
}
