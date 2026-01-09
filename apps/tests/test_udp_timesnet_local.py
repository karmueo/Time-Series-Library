import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.feature_config import get_feature_cols
from apps.udp_timesnet_predict import load_local_trajectory


class LocalTrajectoryLoadTest(unittest.TestCase):
    def test_load_local_xls(self):
        cols = get_feature_cols()
        header = "\t".join(cols)
        values = [
            "1000", "1", "2", "2000", "3", "4",
            "500", "600", "7", "8", "9", "10", "11", "12",
        ]
        content = f"{header}\n" + "\t".join(values) + "\n"

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "sample.xls"
            file_path.write_text(content, encoding="gbk")

            rows = load_local_trajectory(str(file_path), 1, cols)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].shape[0], len(cols))
            self.assertAlmostEqual(rows[0][0], 1.0, places=6)
            self.assertAlmostEqual(rows[0][3], 2.0, places=6)
            self.assertAlmostEqual(rows[0][6], 0.5, places=6)
            self.assertAlmostEqual(rows[0][7], 0.6, places=6)


if __name__ == "__main__":
    unittest.main()
