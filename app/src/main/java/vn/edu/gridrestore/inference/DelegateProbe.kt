package vn.edu.gridrestore.inference

import org.tensorflow.lite.Delegate
import org.json.JSONObject

/** C-context observation before/after GPU delegation; no graph replacements. */
internal class DelegateProbe(target: Long) : Delegate {
    private var pointer = create(target)
    override fun getNativeHandle(): Long = handle(pointer)
    fun evidence(): JSONObject = JSONObject(snapshot(pointer))
    override fun close() { if (pointer != 0L) { destroy(pointer); pointer = 0 } }
    private external fun create(target: Long): Long
    private external fun handle(pointer: Long): Long
    private external fun snapshot(pointer: Long): String
    private external fun destroy(pointer: Long)
    companion object { init { System.loadLibrary("gridrestore_probe") } }
}
