package vn.edu.gridrestore

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assume.assumeTrue
import org.junit.Assert.assertEquals
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import vn.edu.gridrestore.data.BundledModels
import vn.edu.gridrestore.data.Experiments
import java.io.File

/** Opt-in physical performance run, independent of dataset evaluation. */
@RunWith(AndroidJUnit4::class)
class BenchmarkDeviceTest {
    @Test fun benchmarkOnlyWhenExplicitlyRequested() {
        assumeTrue("Timing is opt-in",InstrumentationRegistry.getArguments().getString("runBenchmark")=="true")
        val instrumentation=InstrumentationRegistry.getInstrumentation()
        val context=instrumentation.targetContext
        fun progress(message:String) {
            File(context.filesDir,"device-progress.txt").writeText(message)
            instrumentation.sendStatus(1,Bundle().apply { putString("stream","$message\n") })
        }
        BundledModels.installMissing(context,::progress)
        val input=File(context.filesDir,"input.png")
        val source=if (input.isFile) requireNotNull(BitmapFactory.decodeFile(input.path))
            else context.assets.open("sample-input.png").use { requireNotNull(BitmapFactory.decodeStream(it)) }
        try {
            if (!input.exists()) input.outputStream().use { source.compress(Bitmap.CompressFormat.PNG,100,it) }
            val result=Experiments.benchmark(context.filesDir,source,::progress,context)
            File(context.filesDir,"benchmark.json").writeText(result.toString(2))
            val cpu=result.getJSONArray("cases").getJSONObject(0)
            assertEquals("measured",cpu.getString("status"))
            assertEquals(5,cpu.getInt("warmups"));assertEquals(50,cpu.getJSONArray("samplesMs").length())
            progress("DONE: CPU/GPU benchmark; inspect GPU availability/classification")
        } finally { source.recycle() }
    }
}
