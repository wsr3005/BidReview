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
        self.assertEqual(amount.get("field"), "bid_bond")

        term = next(c for c in constraints if c.get("type") == "term")
        self.assertEqual(term.get("op"), "<=")
        self.assertEqual(term.get("value"), 30)
        self.assertEqual(term.get("unit"), "天")

        qty = next(c for c in constraints if c.get("type") == "quantity")
        self.assertEqual(qty.get("op"), ">=")
        self.assertEqual(qty.get("value"), 2)
        self.assertEqual(qty.get("unit"), "份")

    def test_quantity_without_intent_or_op_is_ignored(self) -> None:
        text = "第2章 5项评分因素详见附表。"
        constraints = extract_constraints(text)
        self.assertFalse(any(c.get("type") == "quantity" for c in constraints))

    def test_quantity_with_intent_is_extracted_without_op(self) -> None:
        text = "投标文件提供2份纸质版。"
        constraints = extract_constraints(text)
        qty = next(c for c in constraints if c.get("type") == "quantity")
        self.assertIsNone(qty.get("op"))
        self.assertEqual(qty.get("value"), 2)
        self.assertEqual(qty.get("unit"), "份")

    def test_amount_fields_are_disambiguated_in_mixed_clause(self) -> None:
        text = "本项目最高限价为100万元，投标保证金为2万元。"
        constraints = [c for c in extract_constraints(text) if c.get("type") == "amount"]
        self.assertEqual(len(constraints), 2)
        by_field = {c.get("field"): c for c in constraints}
        self.assertIn("ceiling_price", by_field)
        self.assertIn("bid_bond", by_field)
        self.assertEqual(by_field["ceiling_price"].get("value_fen"), 100 * 10000 * 100)
        self.assertEqual(by_field["bid_bond"].get("value_fen"), 2 * 10000 * 100)


if __name__ == "__main__":
    unittest.main()
