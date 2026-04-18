"""OverWorld plugin for Daydream Scope."""

import scope.core

from .pipeline import Waypoint360Pipeline, WaypointPipeline


@scope.core.hookimpl
def register_pipelines(register):
    register(WaypointPipeline)
    register(Waypoint360Pipeline)


__all__ = ["Waypoint360Pipeline", "WaypointPipeline"]
