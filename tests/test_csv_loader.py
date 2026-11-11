import tempfile
import unittest
from pathlib import Path

from core.csv_loader import CSVLoadError, CSVOptions, load_csv, load_preview, sniff_options
from core.dtypes import ColumnType, date_to_int, infer_type

import pandas as pd


class TestCsvLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text, encoding="utf-8"):
        p = self.dir / name
        p.write_text(text, encoding=encoding)
        return p

    def test_sniff_defaults_utf8_comma(self):
        p = self.write("a.csv", "x,y\n1,2\n3,4\n")
        opts = sniff_options(p)
        self.assertEqual(opts.encoding, "utf-8")
        self.assertEqual(opts.delimiter, ",")
        self.assertEqual(opts.decimal, ".")

    def test_sniff_semicolon_decimal_comma(self):
        p = self.write("b.csv", "x;y\n1,5;2\n3,25;4\n", encoding="cp1251")
        opts = sniff_options(p)
        self.assertEqual(opts.delimiter, ";")
        self.assertEqual(opts.decimal, ",")

    def test_missing_file(self):
        with self.assertRaises(CSVLoadError):
            sniff_options(self.dir / "nope.csv")

    def test_empty_file(self):
        p = self.write("empty.csv", "")
        with self.assertRaises(CSVLoadError):
            sniff_options(p)

    def test_preview_20_rows(self):
        rows = "a,b\n" + "\n".join(f"{i},cat{i % 3}" for i in range(100))
        p = self.write("big.csv", rows)
        df, types = load_preview(p, CSVOptions(), n_rows=20)
        self.assertEqual(len(df), 20)
        self.assertEqual(types["a"], ColumnType.NUMERIC)
        self.assertEqual(types["b"], ColumnType.CATEGORICAL)

    def test_load_date_to_int(self):
        p = self.write("d.csv", "d,x\n2025-12-29,1\n2026-01-01,2\n")
        df, types = load_csv(p, CSVOptions())
        self.assertEqual(types["d"], ColumnType.DATE)
        self.assertEqual(int(df["d"].iloc[0]), 20251229)
        self.assertEqual(int(df["d"].iloc[1]), 20260101)

    def test_type_override(self):
        p = self.write("o.csv", "x\n1\n2\n")
        df, types = load_csv(p, CSVOptions(), {"x": ColumnType.CATEGORICAL})
        self.assertEqual(types["x"], ColumnType.CATEGORICAL)
        self.assertTrue(isinstance(df["x"].dtype, pd.StringDtype))

    def test_date_int_conversion(self):
        s = pd.Series(["2025-12-29", "bad", None])
        out = date_to_int(s)
        self.assertEqual(int(out.iloc[0]), 20251229)
        self.assertTrue(out.iloc[1] is pd.NA or pd.isna(out.iloc[1]))

    def test_infer_bool_is_categorical(self):
        self.assertEqual(infer_type(pd.Series([True, False])), ColumnType.CATEGORICAL)


if __name__ == "__main__":
    unittest.main()
