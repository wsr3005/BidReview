from __future__ import annotations

import unittest

from bidagent.constraints import extract_constraints


class ConstraintExtractionTests(unittest.TestCase):
    def test_extracts_amount_term_quantity(self) -> None:
        text = "投标保证金不少于10万元；工期不超过30天；提供至少2份复印件。"
        constraints = extract_constraints(text)
        types = {c.get("type") for c in constraints}
        self.assertIn("amount", types)
        self.assertIn("term", types)
        self.assertIn("quantity", types)

        amount = next(c for c in constraints if c.get("type") == "amount")
        self.assertEqual(amount.get("op"), ">=")
        self.assertEqual(amount.get("value_fen"), 10 * 10000 * 100)

        term = next(c for c in constraints if c.get("type") == "term")
        self.assertEqual(term.get("op"), "<=")
        self.assertEqual(term.get("value"), 30)
        self.assertEqual(term.get("unit"), "天")

        qty = next(c for c in constraints if c.get("type") == "quantity")
        self.assertEqual(qty.get("op"), ">=")
        self.assertEqual(qty.get("value"), 2)
        self.assertEqual(qty.get("unit"), "份")


if __name__ == "__main__":
    unittest.main()

