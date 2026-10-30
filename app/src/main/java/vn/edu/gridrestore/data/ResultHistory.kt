package vn.edu.gridrestore.data

import java.io.File

/** Preserve previous evidence while preventing stale metrics from appearing current. */
object ResultHistory {
    fun archive(filesDir: File, vararg names: String) {
        val existing=names.map { File(filesDir,it) }.filter { it.isFile }
        if (existing.isEmpty()) return
        val history=File(filesDir,"history/${System.currentTimeMillis()}-${System.nanoTime()}")
        check(history.mkdirs()) { "Không thể lưu lịch sử báo cáo" }
        existing.forEach { check(it.renameTo(File(history,it.name))) { "Không thể lưu lịch sử ${it.name}" } }
    }
}
