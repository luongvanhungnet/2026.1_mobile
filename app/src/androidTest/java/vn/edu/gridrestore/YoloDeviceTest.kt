package vn.edu.gridrestore

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assert.*
import org.json.JSONObject
import org.json.JSONArray
import vn.edu.gridrestore.core.EvaluationTask
import vn.edu.gridrestore.data.*
import vn.edu.gridrestore.inference.*
import java.io.File
import java.io.FileInputStream
import java.nio.channels.FileChannel
import org.tensorflow.lite.Interpreter

@RunWith(AndroidJUnit4::class)
class YoloDeviceTest {
    @Test fun nativeObserverPreservesCpuOutputAndRecordsPlan() {
        val context=InstrumentationRegistry.getInstrumentation().targetContext
        val model=File(context.filesDir,"models/yolo_detect.tflite")
        val mapped=FileInputStream(model).use { it.channel.map(FileChannel.MapMode.READ_ONLY,0,model.length()) }
        val before=DelegateProbe(0); val after=DelegateProbe(0)
        var baseline:FloatArray?=null
        try {
            repeat(2) { pass ->
                val options=Interpreter.Options().setNumThreads(4).setUseXNNPACK(false)
                if (pass==1) { options.addDelegate(before);options.addDelegate(after) }
                Interpreter(mapped,options).use { interpreter ->
                    val input=floatBuffer(interpreter.getInputTensor(0).numElements())
                    val output=floatBuffer(interpreter.getOutputTensor(0).numElements())
                    interpreter.run(input,output)
                    val values=FloatArray(output.capacity()/4) { output.getFloat(it*4) }
                    if (pass==0) baseline=values else {
                        assertEquals(baseline!!.size,values.size)
                        values.indices.forEach { assertEquals(baseline!![it],values[it],1e-6f) }
                    }
                }
            }
            val a=before.evidence();val b=after.evidence()
            assertTrue(a.getBoolean("prepared"));assertTrue(b.getBoolean("prepared"))
            assertTrue(a.getInt("executionNodes")>0)
            assertEquals(a.getInt("executionNodes"),b.getInt("cpuNodes"))
            File(context.filesDir,"native-probe-validation.json").writeText(JSONObject().put("before",a).put("after",b).put("cpuOutputPreserved",true).toString(2))
        } finally { before.close();after.close() }
    }

    @Test fun runRealYoloAndInspectGpuPlan() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val files = context.filesDir
        val fixtureRoot=File(context.cacheDir,"yolo-validation").apply { mkdirs() }
        val dataset = File("/data/local/tmp/gridrestore-test.zip").inputStream().use { DatasetImporter.importZip(it,fixtureRoot) }
        val source = dataset.samples.first().bitmap()
        val result = JSONObject().put("device", Experiments.device())
        try {
            listOf(EvaluationTask.DETECTION, EvaluationTask.SEGMENTATION).forEach { task ->
                val name = if (task == EvaluationTask.DETECTION) "yolo_detect" else "yolo_segment"
                val model = File(files,"models/$name.tflite")
                val predictions = Yolo(model,task).use { it.predict(source,dataset.samples.first().id) }
                assertFalse("Known labelled fixture should produce YOLO predictions",predictions.isEmpty())
                result.put(name,JSONObject().put("sha256",Experiments.hash(model)).put("predictions",JSONArray(predictions.map {
                    JSONObject().put("classId",it.classId).put("confidence",it.confidence.toDouble())
                        .put("box",JSONArray(listOf(it.box.left,it.box.top,it.box.right,it.box.bottom)))
                        .put("maskPixels",it.mask?.pixels?.count { pixel -> pixel } ?: JSONObject.NULL)
                })))
                assertTrue("Real YOLO must produce finite probabilities", predictions.all { it.confidence.isFinite() })
                if (task == EvaluationTask.SEGMENTATION) assertTrue(predictions.all { it.mask != null })
            }
            val gpu = JSONObject()
            try {
                TfliteSession(File(files,"models/yolo_detect.tflite"),Backend.GPU,4).use { session ->
                    val interpreter = session.interpreter
                    val input = floatBuffer(interpreter.getInputTensor(0).numElements())
                    val output = (0 until interpreter.outputTensorCount).associateWith { floatBuffer(interpreter.getOutputTensor(it).numElements()) as Any }
                    interpreter.runForMultipleInputsOutputs(arrayOf(input),output)
                    gpu.put("status","invoked").put("placement",session.placement())
                    assertNotEquals("UNVERIFIED",session.placement().getString("classification"))
                }
            } catch (error: Exception) { gpu.put("status","unavailable").put("error",error.stackTraceToString()) }
            result.put("gpuProbe",gpu)
            File(files,"yolo-device-validation.json").writeText(result.toString(2))
        } finally { source.recycle() }
    }
}
