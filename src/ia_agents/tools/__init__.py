"""Registered, validated mock ticket tools; no external side effects."""

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field


class TicketArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    ticket_id: str = Field(pattern=r"^T-\d{3}$")


class EscalationArguments(TicketArguments):
    reason: str = Field(min_length=1, max_length=500)


TICKETS = {"T-001": "In progress", "T-002": "Resolved", "T-003": "Waiting for customer"}


def get_ticket_status(ticket_id):
    return {"ticket_id": ticket_id, "status": TICKETS.get(ticket_id, "Not found")}


def escalate_ticket(ticket_id, reason):
    return {"ticket_id": ticket_id, "reason": reason,
            "status": "Escalated (mock)" if ticket_id in TICKETS else "Not found"}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    arguments: type[BaseModel]
    function: Callable

    @property
    def schema(self):
        return self.arguments.model_json_schema()

    def execute(self, arguments):
        validated = self.arguments.model_validate(arguments)
        return self.function(**validated.model_dump())


TICKET_TOOLS = (
    Tool("get_ticket_status", "Look up a mock support ticket by ID.", TicketArguments, get_ticket_status),
    Tool("escalate_ticket", "Simulate escalating a mock ticket with a reason; no real action.",
         EscalationArguments, escalate_ticket),
)


TOOL_GROUPS = {"tickets": TICKET_TOOLS}


def select_tools(groups):
    """Combine named groups, preserving order and avoiding duplicate tools."""
    selected = {}
    for group in groups:
        if group not in TOOL_GROUPS:
            raise ValueError(f"Unknown tool group '{group}'.")
        for tool in TOOL_GROUPS[group]:
            if tool.name in selected and selected[tool.name] != tool:
                raise ValueError(f"Conflicting definitions for tool '{tool.name}'.")
            selected[tool.name] = tool
    if not selected:
        raise ValueError("Select at least one tool group containing tools.")
    return tuple(selected.values())
