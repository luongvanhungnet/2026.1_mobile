package vn.edu.gridrestore.data

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import org.json.JSONObject
import vn.edu.gridrestore.core.BinaryMask
import vn.edu.gridrestore.core.Box
import vn.edu.gridrestore.core.GroundTruth
import java.io.File
import java.io.InputStream
import java.util.zip.ZipInputStream

data class Sample(val id: String, val degradation: String, val file: File, val annotations: List<Annotation>) {
    fun bitmap(): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.path, bounds)
        require(bounds.outWidth > 0 && bounds.outHeight > 0) { "Không đọc được ảnh $id" }
        require(bounds.outWidth.toLong() * bounds.outHeight <= 1_048_576) { "$id vượt 1 megapixel; hãy chuẩn bị ảnh test nhỏ hơn cùng nhãn tương ứng." }
        return requireNotNull(BitmapFactory.decodeFile(file.path))
    }
    fun groundTruth(width: Int, height: Int, masks: Boolean): List<GroundTruth> = annotations.map { annotation ->
        require(annotation.box.left >= 0 && annotation.box.top >= 0 && annotation.box.right <= width && annotation.box.bottom <= height) { "Box ngoài ảnh $id" }
        val mask = if (masks) {
            require(annotation.polygons.isNotEmpty()) { "Thiếu polygon segmentation: $id / class ${annotation.classId}" }
            require(annotation.polygons.flatten().all { (x, y) -> x <= width && y <= height }) { "Polygon ngoài ảnh $id" }
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)
            val paint = Paint().apply { color = Color.WHITE; isAntiAlias = false }
            annotation.polygons.forEach { points ->
                val path = Path()
                path.moveTo(points[0].first, points[0].second)
                points.drop(1).forEach { path.lineTo(it.first, it.second) }
                path.close()
                canvas.drawPath(path, paint)
            }
            val pixels = IntArray(width * height)
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
            bitmap.recycle()
            BinaryMask(width, height, BooleanArray(pixels.size) { pixels[it] != 0 })
        } else null
        GroundTruth(id, annotation.classId, annotation.box, mask)
    }
}

data class Annotation(val classId: Int, val box: Box, val polygons: List<List<Pair<Float, Float>>>)
data class Dataset(val name: String, val samples: List<Sample>, val provenance: JSONObject = JSONObject()) {
    val degradations get() = samples.map { it.degradation }.distinct()
    companion object {
        fun read(directory: File): Dataset {
            val manifest = File(directory, "manifest.json")
            require(manifest.isFile && manifest.length() <= 4_194_304) { "Cần manifest.json ở gốc ZIP (tối đa 4 MB)." }
            val json = JSONObject(manifest.readText())
            require(json.getInt("schemaVersion") == 1) { "schemaVersion phải là 1" }
            val entries = json.getJSONArray("images")
            require(entries.length() in 1..200) { "Bộ test cần 1–200 ảnh" }
            val ids = mutableSetOf<String>()
            val samples = (0 until entries.length()).map { i ->
                val entry = entries.getJSONObject(i)
                val id = entry.getString("id")
                require(id.isNotBlank() && ids.add(id)) { "ID ảnh trống hoặc bị trùng: $id" }
                val category = entry.getString("degradation")
                require(category.isNotBlank()) { "Thiếu loại suy thoái: $id" }
                val image = safeChild(directory, entry.getString("file"))
                require(image.isFile) { "Không tìm thấy ảnh: ${image.name}" }
                val labels = entry.getJSONArray("annotations")
                require(labels.length() <= 200) { "Tối đa 200 nhãn mỗi ảnh" }
                val annotations = (0 until labels.length()).map { j ->
                    val label = labels.getJSONObject(j)
                    val box = label.getJSONArray("box")
                    require(box.length() == 4) { "Box cần [x1,y1,x2,y2] pixel" }
                    val polygons = label.optJSONArray("polygons")
                    val shapes = if (polygons == null) emptyList() else (0 until polygons.length()).map { k ->
                        val shape = polygons.getJSONArray(k)
                        require(shape.length() >= 3) { "Polygon cần ít nhất 3 điểm" }
                        (0 until shape.length()).map { p ->
                            val point = shape.getJSONArray(p)
                            val x = point.getDouble(0).toFloat(); val y = point.getDouble(1).toFloat()
                            require(x.isFinite() && y.isFinite() && x >= 0 && y >= 0) { "Điểm polygon không hợp lệ" }
                            x to y
                        }
                    }
                    val classId = label.getInt("classId")
                    require(classId >= 0) { "classId phải không âm" }
                    val rectangle = Box(box.getDouble(0).toFloat(), box.getDouble(1).toFloat(), box.getDouble(2).toFloat(), box.getDouble(3).toFloat())
                    require(rectangle.right > rectangle.left && rectangle.bottom > rectangle.top) { "Box ground truth phải có diện tích dương" }
                    Annotation(classId, rectangle, shapes)
                }
                Sample(id, category, image, annotations)
            }
            return Dataset(json.optString("name", "Bộ test"), samples, json.optJSONObject("provenance") ?: JSONObject())
        }
    }
}

internal fun safeChild(root: File, path: String): File {
    require(path.isNotBlank() && !path.contains('\\') && !path.contains(':')) { "Đường dẫn ZIP không hợp lệ" }
    val child = File(root, path).canonicalFile
    require(child.path.startsWith(root.canonicalPath + File.separator)) { "Đường dẫn vượt thư mục bộ test" }
    return child
}

object DatasetImporter {
    fun importZip(input: InputStream, filesDir: File): Dataset {
        val staging = File(filesDir, "dataset-staging-${System.nanoTime()}").apply { mkdirs() }
        try {
            var count = 0
            var total = 0L
            val paths = mutableSetOf<String>()
            ZipInputStream(input.buffered()).use { zip ->
                while (true) {
                    val entry = zip.nextEntry ?: break
                    require(++count <= 512) { "ZIP vượt 512 mục" }
                    val destination = safeChild(staging, entry.name)
                    require(paths.add(destination.path)) { "Đường dẫn trùng trong ZIP: ${entry.name}" }
                    if (entry.isDirectory) destination.mkdirs() else {
                        destination.parentFile?.mkdirs()
                        var size = 0L
                        destination.outputStream().use { output ->
                            val buffer = ByteArray(32_768)
                            while (true) {
                                val length = zip.read(buffer)
                                if (length < 0) break
                                size += length; total += length
                                require(size <= 67_108_864 && total <= 536_870_912) { "ZIP vượt giới hạn 64 MB/file hoặc 512 MB tổng" }
                                output.write(buffer, 0, length)
                            }
                        }
                    }
                    zip.closeEntry()
                }
            }
            Dataset.read(staging)
            val target = File(filesDir, "dataset")
            val previous = File(filesDir, "dataset-previous")
            previous.deleteRecursively()
            if (target.exists()) check(target.renameTo(previous)) { "Không thể thay bộ test cũ" }
            if (!staging.renameTo(target)) {
                previous.renameTo(target)
                error("Không thể lưu bộ test mới")
            }
            previous.deleteRecursively()
            ResultHistory.archive(filesDir,"evaluation.json")
            return Dataset.read(target)
        } finally { staging.deleteRecursively() }
    }
}
