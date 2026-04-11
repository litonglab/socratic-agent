from __future__ import annotations

from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from . import services

mcp = FastMCP("networking-lab-agent")


@mcp.tool()
def retrieve_course_docs(
    query: str,
    category: Optional[str] = None,
    hint_level: int = 0,
    max_sources: int = 6,
) -> Dict[str, Any]:
    """Retrieve course document evidence for a query.

    Args:
        query: User question or retrieval query.
        category: Optional question category used by adaptive retrieval.
        hint_level: Optional hint level that affects retrieval breadth.
        max_sources: Maximum number of cited chunks to return.
    """
    return services.retrieve_course_docs(
        query=query,
        category=category,
        hint_level=hint_level,
        max_sources=max_sources,
    )


@mcp.tool()
def get_topology_context(
    query: str,
    experiment_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Load approved topology context for a query or experiment.

    Args:
        query: User query used to infer the experiment when needed.
        experiment_id: Optional explicit experiment id such as lab13.
    """
    return services.get_topology_context(
        query=query,
        experiment_id=experiment_id,
    )


@mcp.tool()
def list_available_experiments() -> Dict[str, Any]:
    """List experiment ids that currently have approved topology data."""
    return services.list_available_experiments()


@mcp.tool()
def get_experiment_manifest(experiment_id: str) -> Dict[str, Any]:
    """Return manifest data for an experiment.

    Args:
        experiment_id: Experiment id such as lab13.
    """
    return services.get_experiment_manifest(experiment_id=experiment_id)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
