package vn.edu.gridrestore.data

import android.graphics.Bitmap
import android.os.Build
import android.os.PowerManager
import android.os.BatteryManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.app.ActivityManager
import org.json.JSONArray
import org.json.JSONObject
import vn.edu.gridrestore.core.EvaluationAccumulator
import vn.edu.gridrestore.core.EvaluationResult
import vn.edu.gridrestore.core.EvaluationTask
import vn.edu.gridrestore.inference.Backend
import vn.edu.gridrestore.inference.Restorer
import vn.edu.gridrestore.inference.Yolo
import java.io.File
import java.security.MessageDigest
import java.time.Instant

object Experiments {
    fun device() = JSONObject().put("manufacturer", Build.MANUFACTURER).put("model", Build.MODEL)
        .put("sdk", Build.VERSION.SDK_INT).put("abis", JSONArray(Build.SUPPORTED_ABIS.toList()))
        .put("hardware",Build.HARDWARE).put("fingerprint",Build.FINGERPRINT)
        .put("isEmulator",Build.FINGERPRINT.startsWith("generic") || Build.MODEL.contains("SDK",ignoreCase=true) || Build.HARDWARE.contains("ranchu"))

    private fun environment(context: Context?): JSONObject {
        val result=JSONObject()
        if (context==null) return result.put("status","unavailable")
        val battery=context.registerReceiver(null,IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val temperature=battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE,-1) ?: -1
        result.put("batteryTemperatureC",if (temperature>=0) temperature/10.0 else JSONObject.NULL)
        result.put("batteryLevel",battery?.getIntExtra(BatteryManager.EXTRA_LEVEL,-1) ?: JSONObject.NULL)
        result.put("batteryScale",battery?.getIntExtra(BatteryManager.EXTRA_SCALE,-1) ?: JSONObject.NULL)
        result.put("plugged",battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED,-1) ?: JSONObject.NULL)
        if (Build.VERSION.SDK_INT>=29) result.put("thermalStatus",context.getSystemService(PowerManager::class.java).currentThermalStatus)
        val memory=ActivityManager.MemoryInfo()
        context.getSystemService(ActivityManager::class.java).getMemoryInfo(memory)
        return result.put("availableRamBytes",memory.availMem).put("totalRamBytes",memory.totalMem).put("lowMemory",memory.lowMemory)
    }

    fun hash(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { stream ->
            val bytes = ByteArray(65536)
            while (true) { val count = stream.read(bytes); if (count < 0) break; digest.update(bytes, 0, count) }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    fun modelInfo(filesDir: File) = JSONObject().also { result ->
        listOf("gridformer", "yolo_detect", "yolo_segment").forEach { name ->
            val file = File(filesDir, "models/$name.tflite")
            if (file.exists()) result.put(name, JSONObject().put("sha256", hash(file)).put("bytes", file.length()))
        }
    }

    private fun metrics(result: EvaluationResult) = JSONObject()
        .put("mAP50", result.map50 ?: JSONObject.NULL).put("mAP50_95", result.map50To95 ?: JSONObject.NULL)
        .put("predictions", result.predictionCount).put("groundTruth", result.groundTruthCount)
        .put("notes", JSONArray(result.notes)).put("perClass", JSONArray(result.perClass.map {
            JSONObject().put("classId", it.classId).put("AP50", it.ap50).put("AP50_95", it.ap50To95)
                .put("groundTruth", it.groundTruthCount).put("predictions", it.predictionCount)
        }))

    private class PairMetrics(task: EvaluationTask) {
        val before = EvaluationAccumulator(task)
        val after = EvaluationAccumulator(task)
        fun json(): JSONObject {
            val a = before.result(); val b = after.result()
            return JSONObject().put("before", metrics(a)).put("after", metrics(b))
                .put("delta_mAP50", if (a.map50 != null && b.map50 != null) b.map50 - a.map50 else JSONObject.NULL)
                .put("delta_mAP50_95", if (a.map50To95 != null && b.map50To95 != null) b.map50To95 - a.map50To95 else JSONObject.NULL)
        }
    }

    fun evaluate(filesDir: File, dataset: Dataset, progress: (String) -> Unit): JSONObject {
        val detection = (dataset.degradations + "ALL").associateWith { PairMetrics(EvaluationTask.DETECTION) }
        val segmentation = (dataset.degradations + "ALL").associateWith { PairMetrics(EvaluationTask.SEGMENTATION) }
        require("ALL" !in dataset.degradations) { "Tên suy thoái ALL được dành cho báo cáo tổng" }
        val outputDir = File(filesDir, "evaluation-images").apply { mkdirs() }
        val started = System.nanoTime()
        Restorer(File(filesDir, "models/gridformer.tflite")).use { restorer ->
            Yolo(File(filesDir, "models/yolo_detect.tflite"), EvaluationTask.DETECTION).use { detector ->
                Yolo(File(filesDir, "models/yolo_segment.tflite"), EvaluationTask.SEGMENTATION).use { segmenter ->
                    dataset.samples.forEachIndexed { index, sample ->
                        if (Thread.currentThread().isInterrupted) throw InterruptedException("Đã hủy đánh giá")
                        progress("${index + 1}/${dataset.samples.size} · ${sample.id} · Khôi phục")
                        val source = sample.bitmap()
                        require(source.width.toLong() * source.height * (sample.annotations.size + 100) <= minOf(67_108_864L, Runtime.getRuntime().maxMemory() / 4)) {
                            "${sample.id}: mask có thể vượt RAM thiết bị. Giảm kích thước ảnh test và scale lại nhãn trước khi nhập."
                        }
                        var restored: Bitmap? = null
                        try {
                            restored = restorer.restore(source)
                            val keys = listOf(sample.degradation, "ALL")
                            progress("${index + 1}/${dataset.samples.size} · Detection trước/sau")
                            val boxLabels = sample.groundTruth(source.width, source.height, false)
                            require(boxLabels.all { it.classId < detector.classCount && it.classId < segmenter.classCount }) { "classId nhãn vượt số lớp của mô hình YOLO" }
                            fun addBoxes(bitmap: Bitmap, after: Boolean) {
                                val predictions = detector.predict(bitmap, sample.id)
                                keys.forEach { key -> (if (after) detection.getValue(key).after else detection.getValue(key).before).addImage(sample.id, predictions, boxLabels) }
                            }
                            addBoxes(source, false); addBoxes(restored, true)
                            progress("${index + 1}/${dataset.samples.size} · Segmentation trước/sau")
                            val maskLabels = sample.groundTruth(source.width, source.height, true)
                            fun addMasks(bitmap: Bitmap, after: Boolean) {
                                val predictions = segmenter.predict(bitmap, sample.id)
                                keys.forEach { key -> (if (after) segmentation.getValue(key).after else segmentation.getValue(key).before).addImage(sample.id, predictions, maskLabels) }
                            }
                            addMasks(source, false); addMasks(restored, true)
                            File(outputDir, "sample_${index + 1}.png").outputStream().use { restored.compress(Bitmap.CompressFormat.PNG, 100, it) }
                        } finally { source.recycle(); restored?.recycle() }
                    }
                }
            }
        }
        return JSONObject().put("schemaVersion", 1).put("kind", "evaluation").put("createdAt", Instant.now().toString())
            .put("device", device()).put("models", modelInfo(filesDir)).put("dataset", dataset.name)
            .put("provenance", dataset.provenance)
            .put("imageCount", dataset.samples.size).put("degradations", JSONArray(dataset.degradations))
            .put("manifestSha256", hash(File(filesDir, "dataset/manifest.json")))
            .put("images", JSONArray(dataset.samples.mapIndexed { index, sample ->
                JSONObject().put("id", sample.id).put("degradation", sample.degradation).put("sourceSha256", hash(sample.file))
                    .put("restoredFile", "evaluation-images/sample_${index + 1}.png")
            }))
            .put("coverage", JSONObject().put("fiveCategoriesPresent", dataset.degradations.size >= 5)
                .put("atLeastTwoPerCategory", dataset.degradations.all { type -> dataset.samples.count { it.degradation == type } >= 2 })
                .put("scopeConfirmedByInstructor", false))
            .put("backend", "CPU").put("seconds", (System.nanoTime() - started) / 1e9)
            .put("settings", JSONObject().put("confidence", 0.001).put("nmsIoU", 0.7).put("maxDet", 100)
                .put("classMapping", "Annotation classId must match YOLO model class indices; COCO-80 for bundled export recipe")
                .put("annotationSupport", "Instance polygons; no crowd/ignore/area-range COCO extensions"))
            .put("results", JSONArray((dataset.degradations + "ALL").map { key ->
                JSONObject().put("degradation", key).put("detection", detection.getValue(key).json()).put("segmentation", segmentation.getValue(key).json())
            }))
    }

    fun benchmark(filesDir: File, source: Bitmap, progress: (String) -> Unit, context: Context? = null): JSONObject {
        val cases = JSONArray()
        Backend.entries.forEach { backend ->
            val row = JSONObject().put("requestedBackend", backend.name)
                .put("cpuThreads",4).put("cpuXnnpackEnabled",backend==Backend.CPU).put("gpuPrecisionLossAllowed",false)
                .put("environmentStart",environment(context))
            try {
                Restorer(File(filesDir, "models/gridformer.tflite"), backend).use { runner ->
                    val result = runner.benchmark(source) { done, total -> progress("${backend.name} · $done/$total · 5 warm-up + 50 lần đo") }
                    row.put("status", "measured").put("warmups", result.warmups).put("runs", result.runs)
                        .put("meanMs", result.meanMs).put("minMs", result.minMs).put("maxMs", result.maxMs)
                        .put("stdDevMs", result.stdDevMs).put("delegation", result.delegation).put("samplesMs", JSONArray(result.samplesMs))
                        .put("placement", result.placement)
                }
            } catch (error: Exception) {
                if (error is InterruptedException) throw error
                row.put("status", "unavailable").put("reason", error.message ?: error.javaClass.simpleName)
            }
            row.put("environmentEnd",environment(context))
            cases.put(row)
        }
        val gpuCase = cases.getJSONObject(1)
        val classification = gpuCase.optJSONObject("placement")?.optString("classification") ?: "UNAVAILABLE"
        return JSONObject().put("schemaVersion", 1).put("kind", "benchmark").put("createdAt", Instant.now().toString())
            .put("device", device()).put("models", modelInfo(filesDir)).put("input", "Center tile 256x256, FP32, RGB [0,1]; identical CPU/GPU input")
            .put("sourceWidth",source.width).put("sourceHeight",source.height)
            .put("inputPngSha256",File(filesDir,"input.png").let { if (it.isFile) hash(it) else JSONObject.NULL })
            .put("timingScope", "Interpreter.run including output transfer; excludes model loading, image decoding and full-image stitching")
            .put("cases", cases).put("gpuClassification", classification)
            .put("gpuFull", if (classification == "GPU_FULL") "measured" else "not_observed")
            .put("gpuMixed", if (classification == "GPU_MIXED") "measured" else "not_observed")
            .put("placementEvidence", "Native observing delegates capture TfLiteContext execution plan before/after GPU delegation; counts must reconcile. Unsupported GPU initialization is recorded separately. A single fixed model/device need not exhibit every possible GPU category.")
    }
}
