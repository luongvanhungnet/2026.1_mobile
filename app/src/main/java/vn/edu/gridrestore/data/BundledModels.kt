package vn.edu.gridrestore.data

import android.content.Context
import java.io.File
import org.json.JSONObject

object BundledModels {
    fun available(context: Context) = context.assets.list("models")?.contains("gridformer.tflite") == true
    fun installMissing(context: Context, progress: (String) -> Unit = {}) {
        if (!available(context)) return
        val names = listOf("gridformer", "yolo_detect", "yolo_segment")
        val metadata = context.assets.open("models/manifest.json").bufferedReader().use { JSONObject(it.readText()) }.getJSONObject("models")
        names.forEach { name ->
            val destination = File(context.filesDir, "models/$name.tflite")
            if (!destination.exists()) {
                progress("Chuẩn bị mô hình $name…")
                destination.parentFile?.mkdirs()
                val staging = File(destination.parentFile, "$name.bundled-pending")
                try {
                    context.assets.open("models/$name.tflite").use { input -> staging.outputStream().use { input.copyTo(it) } }
                    check(Experiments.hash(staging) == metadata.getJSONObject(name).getString("sha256")) { "Sai checksum mô hình $name" }
                    check(staging.renameTo(destination)) { "Không thể cài mô hình $name" }
                } finally { staging.delete() }
            }
        }
    }
}
