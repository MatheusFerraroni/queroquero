import unittest

from queroquero.manifest import preparation_id


class ManifestTest(unittest.TestCase):
    def test_preparation_id_is_stable_and_sensitive_to_source(self) -> None:
        first = preparation_id("a" * 64, "b" * 64, {"kind": "test", "value": 1})
        repeated = preparation_id(
            "a" * 64, "b" * 64, {"kind": "test", "value": 1}
        )
        changed = preparation_id("a" * 64, "b" * 64, {"kind": "test", "value": 2})
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first), 20)


if __name__ == "__main__":
    unittest.main()
