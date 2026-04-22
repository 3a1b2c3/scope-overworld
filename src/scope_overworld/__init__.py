"""OverWorld plugin for Daydream Scope."""

import scope.core

from .pipeline import Waypoint1SmallPipeline, Waypoint360Pipeline, WaypointPipeline


@scope.core.hookimpl
def register_pipelines(register):
    register(WaypointPipeline)
    register(Waypoint360Pipeline)
    register(Waypoint1SmallPipeline)


__all__ = [
    "Waypoint1SmallPipeline",
    "Waypoint360Pipeline",
    "WaypointPipeline",
]
