package vn.edu.gridrestore

import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.ActivityInfo
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*
import org.json.JSONObject
import vn.edu.gridrestore.data.*
import vn.edu.gridrestore.inference.Backend
import vn.edu.gridrestore.inference.Restorer
import vn.edu.gridrestore.inference.Yolo
import vn.edu.gridrestore.core.EvaluationTask
import java.io.File
import java.util.concurrent.Executors

class MainActivity : Activity() {
    private val worker = Executors.newSingleThreadExecutor()
    private val backgroundColor = Color.rgb(16, 24, 32)
    private val surface = Color.rgb(27, 38, 48)
    private val mint = Color.rgb(128, 229, 188)
    private val white = Color.rgb(236, 242, 246)
    private val muted = Color.rgb(159, 178, 188)
    private lateinit var root: LinearLayout
    private lateinit var content: LinearLayout
    private lateinit var status: TextView
    private var selectedTab = 0
    private var busy = false
    private var original: Bitmap? = null
    private var restored: Bitmap? = null
    private var modelTarget = ""
    private var exportFile: File? = null
    private var backend = Backend.CPU
    private val modelNames = listOf("gridformer", "yolo_detect", "yolo_segment")
    private val labels = listOf("GridFormer", "YOLO · Detection", "YOLO · Segmentation")

    override fun onCreate(state: Bundle?) {
        super.onCreate(state)
        window.statusBarColor = backgroundColor
        window.navigationBarColor = backgroundColor
        selectedTab = state?.getInt("tab") ?: 0
        modelTarget = state?.getString("modelTarget") ?: ""
        exportFile = state?.getString("exportFile")?.let { File(it) }
        original = BitmapFactory.decodeFile(File(filesDir, "input.png").path)
        restored = BitmapFactory.decodeFile(File(filesDir, "restored.png").path)
        render()
        if (BundledModels.available(this) && modelNames.any { !modelFile(it).exists() }) {
            runJob("Chuẩn bị mô hình đi kèm") { BundledModels.installMissing(this, ::progress) }
        }
    }

    override fun onSaveInstanceState(state: Bundle) { state.putInt("tab", selectedTab); state.putString("modelTarget", modelTarget); state.putString("exportFile", exportFile?.path); super.onSaveInstanceState(state) }
    override fun onDestroy() { worker.shutdownNow(); super.onDestroy() }
    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()
    private fun shape(color: Int, radius: Int = 18) = GradientDrawable().apply { setColor(color); cornerRadius = dp(radius).toFloat() }
    private fun text(value: String, size: Float = 14f, color: Int = white, bold: Boolean = false) = TextView(this).apply {
        text = value; textSize = size; setTextColor(color); setLineSpacing(dp(3).toFloat(), 1f)
        if (bold) typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
    }
    private fun LinearLayout.addText(value: String, size: Float = 14f, color: Int = white, bold: Boolean = false) = text(value, size, color, bold).also {
        addView(it, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(10) })
    }
    private fun button(parent: LinearLayout, title: String, primary: Boolean = false, action: () -> Unit) = Button(this).apply {
        text = title; isAllCaps = false; textSize = 14f; setTextColor(if (primary) backgroundColor else white)
        background = shape(if (primary) mint else Color.rgb(42, 58, 70), 12)
        minHeight = dp(48); setPadding(dp(12), dp(8), dp(12), dp(8))
        setOnClickListener { action() }; isEnabled = !busy; alpha = if (busy) .5f else 1f
        parent.addView(this, LinearLayout.LayoutParams(-1, -2).apply { topMargin = dp(8); bottomMargin = dp(4) })
    }
    private fun card(title: String, subtitle: String? = null, body: (LinearLayout) -> Unit) {
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; background = shape(surface); setPadding(dp(18), dp(18), dp(18), dp(18))
        }
        content.addView(card, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(14) })
        card.addText(title, 18f, white, true)
        if (subtitle != null) card.addText(subtitle, 13f, muted)
        body(card)
    }

    private fun render() {
        root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(backgroundColor); setPadding(dp(20), dp(12), dp(20), 0) }
        root.setOnApplyWindowInsetsListener { view, insets ->
            if (android.os.Build.VERSION.SDK_INT >= 30) {
                val bars = insets.getInsets(android.view.WindowInsets.Type.systemBars() or android.view.WindowInsets.Type.displayCutout())
                view.setPadding(dp(20) + bars.left, dp(12) + bars.top, dp(20) + bars.right, bars.bottom)
            } else {
                @Suppress("DEPRECATION")
                view.setPadding(dp(20), dp(12) + insets.systemWindowInsetTop, dp(20), insets.systemWindowInsetBottom)
            }
            insets
        }
        setContentView(root)
        root.requestApplyInsets()
        val brand = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
        brand.addView(text("▦", 30f, mint), LinearLayout.LayoutParams(dp(42), dp(44)))
        brand.addView(text("GridRestore", 22f, white, true), LinearLayout.LayoutParams(0, -2, 1f))
        brand.addView(text("BTL  /  AI", 11f, muted))
        root.addView(brand)
        val tabs = LinearLayout(this)
        listOf("Khôi phục", "Đánh giá", "Hiệu năng", "Thiết lập").forEachIndexed { index, title ->
            tabs.addView(text(title, 12f, if (index == selectedTab) backgroundColor else muted, true).apply {
                gravity = Gravity.CENTER; setPadding(0, dp(12), 0, dp(12)); background = shape(if (index == selectedTab) mint else backgroundColor, 10)
                isEnabled = !busy; setOnClickListener { selectedTab = index; render() }
            }, LinearLayout.LayoutParams(0, -2, 1f).apply { topMargin = dp(14); bottomMargin = dp(14) })
        }
        root.addView(tabs)
        status = text(if (busy) "Đang xử lý trên thiết bị…" else "●  OFFLINE  ·  Dữ liệu xử lý trên thiết bị", 12f, mint)
        root.addView(status, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(12) })
        val scroll = ScrollView(this).apply { isFillViewport = true }
        content = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, 0, 0, dp(24)) }
        scroll.addView(content); root.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        when (selectedTab) { 0 -> restoreTab(); 1 -> evaluateTab(); 2 -> benchmarkTab(); else -> setupTab() }
    }

    private fun restoreTab() {
        content.addText("Chi tiết trở lại.\nNgay trên điện thoại.", 30f, white, true)
        content.addText("GridFormer  ·  Khôi phục ảnh suy thoái do thời tiết", 13f, muted)
        val compare = ComparisonView(this).apply { before = original; after = restored }
        content.addView(compare, LinearLayout.LayoutParams(-1, dp(290)).apply { topMargin = dp(8); bottomMargin = dp(12) })
        content.addText(if (restored != null) "Kéo đường chia để so sánh ảnh gốc và ảnh khôi phục." else "Chọn ảnh để bắt đầu. Ảnh được giữ nguyên kích thước.", 12f, muted)
        card("01  /  Ảnh đầu vào", "Chia ô 256×256 · Chồng lấn 32 px · Ghép có trọng số") { box ->
            original?.let { box.addText("${it.width} × ${it.height} px", 13f, mint) }
            button(box, if (original == null) "＋  Chọn ảnh" else "Chọn ảnh khác") { pick(10, "image/*") }
            if (assets.list("")?.contains("sample-input.png") == true) button(box, "Dùng ảnh test đi kèm") {
                runJob("Đọc ảnh mẫu") {
                    ResultHistory.archive(filesDir,"benchmark.json")
                    val bitmap = assets.open("sample-input.png").use { requireNotNull(BitmapFactory.decodeStream(it)) }
                    File(filesDir, "input.png").outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
                    File(filesDir, "restored.png").delete()
                    runOnUiThread { original = bitmap; restored = null }
                }
            }
            val choices = Spinner(this).apply { adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, listOf("CPU", "GPU delegate (có thể kết hợp CPU)")); setSelection(if (backend == Backend.CPU) 0 else 1) }
            box.addView(choices)
            choices.isEnabled = !busy
            choices.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) { backend = if (position == 0) Backend.CPU else Backend.GPU }
                override fun onNothingSelected(parent: AdapterView<*>?) {}
            }
            button(box, "Khôi phục ảnh  →", true) {
                val source = original ?: return@button toast("Chọn ảnh trước")
                runJob("Khôi phục") {
                    val result = Restorer(modelFile("gridformer"), backend).use { it.restore(source) { done, total -> progress("Khôi phục · ô $done/$total") } }
                    File(filesDir, "restored.png").outputStream().use { result.compress(Bitmap.CompressFormat.PNG, 100, it) }
                    runOnUiThread { restored = result }
                }
            }
            if (restored != null) button(box, "Lưu ảnh PNG") { export(File(filesDir, "restored.png"), "image/png", "gridrestore.png") }
        }
        if (!modelFile("gridformer").exists()) card("Cần thêm mô hình", "Mở Thiết lập để nhập GridFormer FP32 TFLite. Quy trình chuyển đổi có trong mã nguồn.") { box ->
            button(box, "Mở thiết lập") { selectedTab = 3; render() }
        }
    }

    private fun evaluateTab() {
        content.addText("Đo chất lượng\ntrước & sau.", 30f, white, true)
        content.addText("Detection và Segmentation là hai phép đánh giá độc lập.", 13f, muted)
        val dataset = runCatching { Dataset.read(File(filesDir, "dataset")) }.getOrNull()
        card("02  /  Bộ ảnh test", "ZIP chứa manifest.json, ảnh suy thoái và nhãn ground truth. Xem docs/dataset.md trong mã nguồn.") { box ->
            box.addText(dataset?.let { "${it.name}\n${it.samples.size} ảnh · ${it.degradations.size} loại suy thoái" } ?: "Chưa có bộ test", 14f, mint)
            if (dataset != null) box.addText(dataset.degradations.joinToString(" · "), 12f, muted)
            button(box, "Nhập bộ test ZIP") { pick(11, "application/zip", arrayOf("application/zip", "application/x-zip-compressed", "application/octet-stream")) }
            if (assets.list("")?.contains("sample-test.zip") == true) button(box, "Dùng 15 ảnh test đi kèm") {
                runJob("Nhập bộ test đi kèm") { assets.open("sample-test.zip").use { DatasetImporter.importZip(it, filesDir) } }
            }
            button(box, "Chạy đánh giá mAP  →", true) {
                val data = dataset ?: return@button toast("Nhập bộ test có nhãn trước")
                runJob("Đánh giá bộ test") {
                    val report = Experiments.evaluate(filesDir, data, ::progress)
                    File(filesDir, "evaluation.json").writeText(report.toString(2))
                }
            }
            box.addText("mAP50 và mAP50:95 tính trên toàn bộ ảnh theo từng class. Nhãn classId phải khớp mô hình YOLO; thiếu nhãn sẽ không có mAP hợp lệ.", 12f, muted)
        }
        val file = File(filesDir, "evaluation.json")
        if (file.exists()) {
            val report = JSONObject(file.readText())
            content.addText("${report.getString("dataset")}\n${report.optString("createdAt")} · ${report.getJSONObject("device").optString("model")}",12f,muted)
            val results = report.getJSONArray("results")
            for (i in 0 until results.length()) {
                val row = results.getJSONObject(i)
                card(row.getString("degradation"), "mAP50:95 · Thang điểm 0–100%") { box ->
                    listOf("detection", "segmentation").forEach { task ->
                        val result = row.getJSONObject(task)
                        val before = result.getJSONObject("before"); val after = result.getJSONObject("after")
                        val delta = if (result.isNull("delta_mAP50_95")) "N/A" else "%+.2f".format(result.getDouble("delta_mAP50_95") * 100) + " điểm %"
                        box.addText("${task.replaceFirstChar { it.uppercase() }}\n${percent(before, "mAP50_95")}  →  ${percent(after, "mAP50_95")}   ($delta)", 14f)
                        box.addText("mAP50: ${percent(before, "mAP50")} → ${percent(after, "mAP50")}", 12f, muted)
                    }
                }
            }
            button(content, "Xuất báo cáo JSON") { export(file, "application/json", "gridrestore-evaluation.json") }
        }
    }

    private fun benchmarkTab() {
        content.addText("Đo trên máy thật.\nGhi nhận rõ ràng.", 30f, white, true)
        card("03  /  Benchmark", "GridFormer · Cùng một ô ảnh 256×256 cho CPU và GPU") { box ->
            box.addText("5  WARM-UP     /     50  LẦN ĐO", 15f, mint, true)
            box.addText("Đo Interpreter.run và đọc đầu ra; không gồm nạp mô hình, giải mã ảnh hay ghép ảnh. Không chạm vào app trong lúc đo.", 13f, muted)
            button(box, "Chạy CPU + thử GPU  →", true) {
                val source = original ?: return@button toast("Chọn một ảnh ở tab Khôi phục trước")
                runJob("Benchmark") { File(filesDir, "benchmark.json").writeText(Experiments.benchmark(filesDir, source, ::progress, this).toString(2)) }
            }
        }
        val file = File(filesDir, "benchmark.json")
        if (file.exists()) {
            val report = JSONObject(file.readText())
            content.addText("${report.optString("createdAt")} · ${report.getJSONObject("device").optString("model")}",12f,muted)
            val cases = report.getJSONArray("cases")
            for (i in 0 until cases.length()) {
                val item = cases.getJSONObject(i)
                card(item.getString("requestedBackend"), item.getString("status")) { box ->
                    if (item.getString("status") == "measured") {
                        box.addText("%.2f ms".format(item.getDouble("meanMs")), 30f, mint, true)
                        box.addText("Min %.2f · Max %.2f · Độ lệch chuẩn %.2f ms".format(item.getDouble("minMs"), item.getDouble("maxMs"), item.getDouble("stdDevMs")), 12f, muted)
                        box.addText(item.getString("delegation"), 12f, muted)
                    } else box.addText(item.optString("reason"), 13f, muted)
                }
            }
            button(content, "Xuất kết quả benchmark") { export(file, "application/json", "gridrestore-benchmark.json") }
        }
        card("Bằng chứng GPU", "Ứng dụng quan sát execution plan trước/sau khi gán GPU. Báo cáo ghi số toán tử GPU và CPU còn lại; chỉ phân loại khi số lượng khớp. Trường hợp GPU không chạy được được lưu kèm lỗi.") {}
    }

    private fun setupTab() {
        content.addText("Sẵn sàng\ncho thí nghiệm.", 30f, white, true)
        card("Mô hình trên thiết bị", "Nhập tệp .tflite FP32 đã chuyển đổi. App xác minh tensor và thử suy luận trước khi lưu mô hình.") { box ->
            modelNames.forEachIndexed { index, name ->
                val file = modelFile(name)
                box.addText("${labels[index]}  ·  ${if (file.exists()) "%.1f MB".format(file.length() / 1048576.0) else "Chưa nhập"}", 14f, if (file.exists()) mint else white, true)
                button(box, "Nhập ${labels[index]}") { modelTarget = name; pick(12, "*/*") }
            }
        }
        card("Nhóm thực hiện", "GridFormer [68]") { box ->
            box.addText("Nguyễn Sỹ Vinh\nNguyễn Phạm Anh Tuấn\nLương Văn Hưng", 14f)
        }
        card("Phạm vi bài tập", "GridFormer gốc xử lý suy thoái thời tiết. Cần thống nhất năm loại test với giảng viên. Không dùng kết quả giả lập để báo cáo mAP hoặc hiệu năng.") { box ->
            box.addText("Android ${android.os.Build.VERSION.RELEASE} · ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}", 12f, muted)
            box.addText("Ứng dụng hoạt động offline. Ảnh, nhãn, mô hình và báo cáo được lưu trong vùng riêng của app; dùng nút xuất để lưu bản sao.", 12f, muted)
        }
    }

    private fun percent(json: JSONObject, key: String) = if (json.isNull(key)) "N/A" else "%.2f%%".format(json.getDouble(key) * 100)
    private fun modelFile(name: String) = File(filesDir, "models/$name.tflite")
    private fun toast(message: String) { Toast.makeText(this, message, Toast.LENGTH_LONG).show() }
    private fun progress(message: String) { runOnUiThread { if (!isDestroyed) status.text = message } }
    private fun runJob(title: String, task: () -> Unit) {
        if (busy) return
        busy = true; requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LOCKED; render(); status.text = title
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        worker.execute {
            var failure: Throwable? = null
            try { task() } catch (error: Throwable) { failure = error }
            runOnUiThread {
                if (!isDestroyed) {
                    busy = false; requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED; render()
                    window.clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    if (failure != null) AlertDialog.Builder(this).setTitle("Chưa thực hiện được").setMessage(failure.message ?: failure.javaClass.simpleName).setPositiveButton("Đóng", null).show()
                    else toast("Đã hoàn thành: $title")
                }
            }
        }
    }

    private fun pick(code: Int, mime: String, types: Array<String>? = null) {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply { type = mime; addCategory(Intent.CATEGORY_OPENABLE); if (types != null) putExtra(Intent.EXTRA_MIME_TYPES, types) }
        startActivityForResult(intent, code)
    }
    private fun export(file: File, mime: String, filename: String) {
        exportFile = file
        startActivityForResult(Intent(Intent.ACTION_CREATE_DOCUMENT).apply { type = mime; addCategory(Intent.CATEGORY_OPENABLE); putExtra(Intent.EXTRA_TITLE, filename) }, 13)
    }
    @Deprecated("Legacy result API for dependency-free Activity")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        when (requestCode) {
            10 -> runJob("Đọc ảnh") {
                val bitmap = ImageFiles.read(this, uri)
                ResultHistory.archive(filesDir,"benchmark.json")
                File(filesDir, "input.png").outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
                File(filesDir, "restored.png").delete()
                runOnUiThread { original = bitmap; restored = null }
            }
            11 -> runJob("Nhập bộ test") { contentResolver.openInputStream(uri).use { DatasetImporter.importZip(requireNotNull(it), filesDir) } }
            12 -> {
                val target = modelTarget
                if (target !in modelNames) return toast("Chọn lại nút nhập mô hình")
                runJob("Kiểm tra $target") {
                    val file = modelFile(target); file.parentFile?.mkdirs()
                    val staging = File(file.parentFile, "$target.pending")
                    try {
                        contentResolver.openInputStream(uri).use { stream ->
                            requireNotNull(stream).use { input -> staging.outputStream().use { out ->
                                val bytes = ByteArray(65536); var total = 0L
                                while (true) { val count = input.read(bytes); if (count < 0) break; total += count; require(total <= 536_870_912) { "Mô hình vượt 512 MB" }; out.write(bytes, 0, count) }
                            } }
                        }
                        val probe = Bitmap.createBitmap(32, 32, Bitmap.Config.ARGB_8888)
                        try {
                            if (target == "gridformer") Restorer(staging).use { it.restore(probe).recycle() }
                            else Yolo(staging, if (target == "yolo_detect") EvaluationTask.DETECTION else EvaluationTask.SEGMENTATION).use { it.predict(probe, "validation") }
                        } finally { probe.recycle() }
                        check(staging.renameTo(file)) { "Không thể lưu mô hình" }
                        if (target=="gridformer") ResultHistory.archive(filesDir,"evaluation.json","benchmark.json")
                        else ResultHistory.archive(filesDir,"evaluation.json")
                    } finally { staging.delete() }
                }
            }
            13 -> {
                val file = exportFile ?: return toast("Chọn xuất lại tệp")
                runJob("Xuất tệp") { contentResolver.openOutputStream(uri).use { output -> file.inputStream().use { it.copyTo(requireNotNull(output)) } } }
            }
        }
    }
}
