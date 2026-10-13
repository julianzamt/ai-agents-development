"""Data contracts for class 3, section 1."""

import json
import unittest

from pydantic import ValidationError

from agentes_ia.schemas import SupportQuery


class SupportQueryTests(unittest.TestCase):
    def setUp(self):
        self.data = dict(category="technical", urgency="high",
                          summary="The service is unresponsive.", requires_escalation=True)

    def test_valid_dict_and_json(self):
        query = SupportQuery.model_validate(self.data)
        self.assertEqual(query.model_dump(), self.data)
        self.assertEqual(SupportQuery.model_validate_json(json.dumps(self.data)), query)

    def test_invalid_data_is_rejected(self):
        cases = [dict(self.data, category="unknown"),
                 dict(self.data, urgency="urgent"),
                 dict(self.data, summary="x" * 201),
                 dict(self.data, summary=""),
                 dict(self.data, requires_escalation="true"),
                 dict(self.data, phone="123"),
                 {k: v for k, v in self.data.items() if k != "requires_escalation"}]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                SupportQuery.model_validate(data)

    def test_schema_declares_required_fields_and_constraints(self):
        schema = SupportQuery.model_json_schema()
        self.assertEqual(set(schema["required"]), set(self.data))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["requires_escalation"]["type"], "boolean")
        self.assertEqual(schema["properties"]["summary"]["maxLength"], 200)


if __name__ == "__main__":
    unittest.main()
