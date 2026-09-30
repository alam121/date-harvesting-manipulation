"""Application planning policy, independent of the cuRobo Python package."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PlanOptions:
    """Policy for one planning request.

    Attributes:
        max_attempts: Upper bound on trajectory-optimization attempts per plan.
            Each attempt runs a full seeded IK solve plus a trajectory
            optimization, so this multiplies real GPU work per plan. cuRobo v2's
            own default is 5.
        graph_attempt: 1-based attempt number at which graph (PRM) seeding becomes
            available, using the convention this policy was written against -- v1
            enabled the graph when ``n >= graph_attempt - 1``. ``None`` means
            "on the last attempt". Call :meth:`enable_graph_attempt` to convert
            this to the 0-based index cuRobo v2 expects.
    """

    max_attempts: int = 5
    graph_attempt: int | None = None

    def enable_graph_attempt(self) -> int:
        """Return cuRobo v2's 0-based ``enable_graph_attempt`` for this policy.

        v2 enables graph seeding when ``current_attempt >= enable_graph_attempt``
        inside ``range(max_attempts)`` (see ``MotionPlanner._plan_pose_single``
        and ``MotionPlanner.plan_cspace``), whereas this policy is 1-based.

        Passing the 1-based v1 number straight through therefore both shifted the
        graph later by one attempt and made the ``None``/"last attempt" case
        unreachable: ``graph_attempt=None`` translated to ``max_attempts``, and
        ``current_attempt`` never reaches ``max_attempts`` inside
        ``range(max_attempts)``. The graph planner was consequently never called
        for any plan configured that way. Clamp rather than allow an unreachable
        value, so a misconfiguration degrades to "graph later" instead of "graph
        never".
        """
        if self.max_attempts <= 1:
            return 0
        if self.graph_attempt is None:
            return self.max_attempts - 1
        return max(0, min(int(self.graph_attempt) - 1, self.max_attempts - 1))

    def graph_on_first_attempt(self) -> bool:
        """True when this policy requests graph seeding from the first attempt."""
        return self.enable_graph_attempt() == 0
