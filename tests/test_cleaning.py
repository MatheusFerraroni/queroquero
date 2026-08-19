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

    def test_brwac_detokenization_fixes_only_targeted_spacing(self) -> None:
        value = (
            "Direitos Humanos , nesta terça-feira ( 8 ) , no ( TJGO ) . "
            "Filhos - Transformando realidades"
        )
        self.assertEqual(clean_text(value), value)
        self.assertEqual(
            clean_text(value, punctuation_spacing="detokenize_brwac_v1"),
            "Direitos Humanos, nesta terça-feira (8), no (TJGO). "
            "Filhos - Transformando realidades",
        )

    def test_unknown_punctuation_spacing_policy_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown punctuation spacing policy"):
            clean_text("texto", punctuation_spacing="unknown")


if __name__ == "__main__":
    unittest.main()
