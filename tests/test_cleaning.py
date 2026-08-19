import unicodedata
import unittest

from queroquero.datasets.base import clean_text


class CleaningTests(unittest.TestCase):
    def test_html_entities_controls_and_normalization_are_conservative(self) -> None:
        value = (
            "<p>A&#x301;rea\u0091 útil</p>"
            "<script>segredo de script</script>"
            "<style>regra oculta</style>"
        )
        cleaned = clean_text(value)
        self.assertEqual(unicodedata.normalize("NFC", cleaned), cleaned)
        self.assertIn("Área útil", cleaned)
        self.assertNotIn("\u0091", cleaned)
        self.assertNotIn("segredo de script", cleaned)
        self.assertNotIn("regra oculta", cleaned)

    def test_malformed_angle_bracket_text_is_not_silently_truncated(self) -> None:
        value = "comparação <nao-fechado continua aqui"
        self.assertEqual(clean_text(value), value)


if __name__ == "__main__":
    unittest.main()
