"""HITL — Human-in-the-Loop approval gates for agent pipelines.

Pause pipeline execution at critical decision points and require human sign-off
before proceeding. Supports CLI interaction, auto-approve timeout, and webhook
callbacks for integration with external systems.

Usage:
    gate = HITLGate(name="deploy_approval", description="Approve before deploy?")
    node = AgentNode(agent=MyAgent(), hitl_gate=gate)

    # Gate blocks until approved:
    orchestrator.add_node(node)
    results = orchestrator.run(context={"action": "deploy to production"})

Integration:
    For non-interactive use, call gate.approve() or gate.reject() from
    a separate thread/process/webhook handler while the pipeline waits.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"
    AUTO_APPROVED = "auto_approved"


@dataclass
class ApprovalRequest:
    """A pending approval request created when an HITL gate is triggered.

    Attributes:
        gate_id: Unique ID for this approval instance.
        gate_name: Human-readable name of the gate.
        description: What is being approved.
        context_snapshot: Pipeline context at the time of the request.
        agent_output: The agent output awaiting approval.
        status: Current approval status.
        approver: Who approved/rejected (populated on resolution).
        notes: Optional approver notes.
        created_at: Unix timestamp of creation.
        resolved_at: Unix timestamp of resolution.
    """

    gate_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    gate_name: str = ""
    description: str = ""
    context_snapshot: dict[str, Any] = field(default_factory=dict)
    agent_output: dict[str, Any] = field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING
    approver: str = ""
    notes: str = ""
    created_at: float = field(default_factory=time.time)
    resolved_at: float | None = None

    def resolve(
        self,
        status: ApprovalStatus,
        approver: str = "",
        notes: str = "",
    ) -> None:
        self.status = status
        self.approver = approver
        self.notes = notes
        self.resolved_at = time.time()

    @property
    def wait_time_seconds(self) -> float:
        end = self.resolved_at or time.time()
        return round(end - self.created_at, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "gate_name": self.gate_name,
            "description": self.description,
            "status": self.status.value,
            "approver": self.approver,
            "notes": self.notes,
            "wait_time_seconds": self.wait_time_seconds,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }


# Type alias for webhook/callback handlers
ApprovalCallback = Callable[[ApprovalRequest], None]


class HITLGate:
    """Human-in-the-loop approval gate.

    Blocks pipeline execution until a human approves, rejects, or the
    auto-approve timeout expires.

    Args:
        name: Unique gate name (shown in logs and dashboards).
        description: What decision is being gated.
        auto_approve_after: Seconds to wait before auto-approving. None = wait forever.
        required: If False, rejection doesn't fail the pipeline.
        interactive: If True, prompt on stdin for CLI usage.
        on_request: Callback fired when a request is created (e.g. send Slack alert).
        on_resolve: Callback fired when a request is resolved.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        *,
        auto_approve_after: float | None = None,
        required: bool = True,
        interactive: bool = True,
        on_request: ApprovalCallback | None = None,
        on_resolve: ApprovalCallback | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.auto_approve_after = auto_approve_after
        self.required = required
        self.interactive = interactive
        self.on_request = on_request
        self.on_resolve = on_resolve

        self._pending: dict[str, ApprovalRequest] = {}
        self._history: list[ApprovalRequest] = []
        self._lock = threading.Lock()

    def request_approval(
        self,
        context: dict[str, Any],
        agent_output: dict[str, Any],
    ) -> ApprovalRequest:
        """Create an approval request and block until resolved.

        Args:
            context: Current pipeline context.
            agent_output: The agent's output data awaiting approval.

        Returns:
            The resolved ApprovalRequest.
        """
        request = ApprovalRequest(
            gate_name=self.name,
            description=self.description,
            context_snapshot=dict(context),
            agent_output=dict(agent_output),
        )

        with self._lock:
            self._pending[request.gate_id] = request

        logger.info(
            f"[HITL:{self.name}] Gate triggered — ID: {request.gate_id}\n"
            f"  Description: {self.description}\n"
            f"  Auto-approve in: {self.auto_approve_after}s"
            if self.auto_approve_after
            else f"[HITL:{self.name}] Gate triggered — ID: {request.gate_id}"
        )

        if self.on_request:
            try:
                self.on_request(request)
            except Exception as e:
                logger.warning(f"[HITL:{self.name}] on_request callback failed: {e}")

        # Resolution strategies (in priority order):
        # 1. Interactive CLI (blocks with input())
        # 2. Auto-approve after timeout
        # 3. Wait indefinitely for external approve()/reject() calls

        if self.interactive:
            self._cli_prompt(request)
        elif self.auto_approve_after is not None:
            self._wait_with_timeout(request)
        else:
            self._wait_indefinitely(request)

        with self._lock:
            self._pending.pop(request.gate_id, None)
            self._history.append(request)

        if self.on_resolve:
            try:
                self.on_resolve(request)
            except Exception as e:
                logger.warning(f"[HITL:{self.name}] on_resolve callback failed: {e}")

        logger.info(
            f"[HITL:{self.name}] Resolved: {request.status.value} "
            f"by '{request.approver}' after {request.wait_time_seconds}s"
        )
        return request

    def approve(self, gate_id: str, approver: str = "system", notes: str = "") -> bool:
        """Approve a pending request (called from external system/webhook).

        Args:
            gate_id: The gate_id from the ApprovalRequest.
            approver: Name/ID of the approver.
            notes: Optional approval notes.

        Returns:
            True if the request was found and approved, False otherwise.
        """
        with self._lock:
            request = self._pending.get(gate_id)
            if not request:
                return False
            request.resolve(ApprovalStatus.APPROVED, approver=approver, notes=notes)
        return True

    def reject(self, gate_id: str, approver: str = "system", reason: str = "") -> bool:
        """Reject a pending request (called from external system/webhook).

        Args:
            gate_id: The gate_id from the ApprovalRequest.
            approver: Name/ID of the rejector.
            reason: Reason for rejection.

        Returns:
            True if the request was found and rejected, False otherwise.
        """
        with self._lock:
            request = self._pending.get(gate_id)
            if not request:
                return False
            request.resolve(ApprovalStatus.REJECTED, approver=approver, notes=reason)
        return True

    def _cli_prompt(self, request: ApprovalRequest) -> None:
        """Interactive CLI prompt for development/demo use."""
        print(f"\n{'='*60}")
        print(f"  HITL APPROVAL GATE: {self.name}")
        print(f"{'='*60}")
        print(f"  Gate ID   : {request.gate_id}")
        print(f"  Description: {self.description}")
        if request.agent_output:
            print(f"  Agent output preview:")
            for k, v in list(request.agent_output.items())[:3]:
                print(f"    {k}: {str(v)[:80]}")
        print(f"{'='*60}")

        if self.auto_approve_after:
            print(f"  [Auto-approves in {self.auto_approve_after}s if no response]")

        while True:
            try:
                answer = input("  Approve? [y/n/details]: ").strip().lower()
            except EOFError:
                # Non-interactive environment: auto-approve
                request.resolve(ApprovalStatus.AUTO_APPROVED, approver="system")
                return

            if answer in ("y", "yes", "approve"):
                notes = ""
                try:
                    notes = input("  Notes (optional): ").strip()
                except EOFError:
                    pass
                request.resolve(
                    ApprovalStatus.APPROVED,
                    approver="cli_user",
                    notes=notes,
                )
                print(f"  ✓ Approved\n{'='*60}\n")
                return
            elif answer in ("n", "no", "reject"):
                reason = ""
                try:
                    reason = input("  Reason (optional): ").strip()
                except EOFError:
                    pass
                request.resolve(
                    ApprovalStatus.REJECTED,
                    approver="cli_user",
                    notes=reason,
                )
                print(f"  ✗ Rejected\n{'='*60}\n")
                return
            elif answer == "details":
                print(f"\n  Full context: {request.context_snapshot}")
                print(f"  Full output: {request.agent_output}\n")
            else:
                print("  Please enter 'y' to approve, 'n' to reject, or 'details'")

    def _wait_with_timeout(self, request: ApprovalRequest) -> None:
        """Block until approved/rejected or timeout expires."""
        deadline = time.time() + (self.auto_approve_after or 0)
        poll_interval = 0.1

        while time.time() < deadline:
            if request.status != ApprovalStatus.PENDING:
                return
            time.sleep(poll_interval)

        # Timed out
        if request.status == ApprovalStatus.PENDING:
            request.resolve(
                ApprovalStatus.AUTO_APPROVED,
                approver="timeout",
                notes=f"Auto-approved after {self.auto_approve_after}s",
            )

    def _wait_indefinitely(self, request: ApprovalRequest) -> None:
        """Block until approve() or reject() is called."""
        while request.status == ApprovalStatus.PENDING:
            time.sleep(0.1)

    @property
    def pending(self) -> list[ApprovalRequest]:
        """All currently pending approval requests."""
        with self._lock:
            return list(self._pending.values())

    @property
    def history(self) -> list[ApprovalRequest]:
        """All resolved approval requests."""
        return list(self._history)

    def approval_rate(self) -> float:
        """Fraction of resolved requests that were approved."""
        resolved = [
            r
            for r in self._history
            if r.status in (ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED)
        ]
        if not self._history:
            return 0.0
        return len(resolved) / len(self._history)

    def summary(self) -> str:
        """Human-readable gate statistics."""
        total = len(self._history)
        approved = sum(
            1
            for r in self._history
            if r.status in (ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED)
        )
        rejected = sum(1 for r in self._history if r.status == ApprovalStatus.REJECTED)
        timed_out = sum(
            1 for r in self._history if r.status == ApprovalStatus.TIMED_OUT
        )
        avg_wait = (
            sum(r.wait_time_seconds for r in self._history) / total if total else 0
        )
        return (
            f"HITLGate '{self.name}' | "
            f"Total: {total} | Approved: {approved} | Rejected: {rejected} | "
            f"Timed out: {timed_out} | Avg wait: {avg_wait:.1f}s"
        )
