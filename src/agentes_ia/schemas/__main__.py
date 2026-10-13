"""Offline demo: python -m agentes_ia.schemas."""

import json

from pydantic import ValidationError

from . import SupportQuery


def main():
    print("JSON Schema:")
    print(json.dumps(SupportQuery.model_json_schema(), indent=2, ensure_ascii=False))
    data = {
        "category": "technical",
        "urgency": "high",
        "summary": "The service has been unresponsive for two hours.",
        "requires_escalation": True,
    }
    query = SupportQuery.model_validate(data)
    print("\nValidated object:", query)
    print("\nObject JSON:", query.model_dump_json(indent=2))

    for name, invalid_data in (
        ("Invalid category", dict(data, category="unknown")),
        ("Missing required field", {k: v for k, v in data.items()
                                    if k != "requires_escalation"}),
    ):
        try:
            SupportQuery.model_validate(invalid_data)
        except ValidationError as error:
            print(f"\n{name} (expected error):\n{error}")


if __name__ == "__main__":
    main()
