package vn.edu.gridrestore.data

import org.junit.Assert.*
import org.junit.Test
import java.io.File

class SafeZipPathTest {
    @Test fun importedZipPathsCannotEscapeDatasetDirectory() {
        val root = File(System.getProperty("java.io.tmpdir"), "gridrestore-path-test")
        assertEquals(File(root, "images/test.png").canonicalFile, safeChild(root, "images/test.png"))
        listOf("../outside.png", "images/../../escape.png", "C:/other/image.png", "images\\test.png").forEach { name ->
            try { safeChild(root, name); fail("Accepted unsafe ZIP path: $name") } catch (_: IllegalArgumentException) { }
        }
    }
}
