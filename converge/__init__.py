"""
Converge — Schedule Risk Analysis Engine.

A Monte Carlo simulation engine for project schedule risk analysis,
with analytical validation and the Risk Driver Method (Hulett, 1996).

Modules
-------
distributions  : Duration probability distributions (Triangular, PERT, Uniform, Normal)
network        : CPM network with forward pass and longest-path computation
sampling       : Pure random and Latin Hypercube Sampling
engine         : Vectorised Monte Carlo orchestrator
results        : Post-processing: percentiles, criticality index, merge bias, sensitivity
risk_drivers   : Risk Driver Method for emergent correlation (Hulett)
calendar       : Working-day to calendar-date conversion
io             : Import/export: CSV, Excel, JSON round-trip
examples       : Built-in example and benchmark networks
"""

from converge.distributions import Triangular, PERT, Uniform, NormalTruncated
from converge.network import (
    Activity, Network, Predecessor,
    NetworkError, CycleError, MissingPredecessorError, InvalidRelationshipError,
    parse_predecessor, parse_predecessors, format_predecessor, format_predecessors,
)
from converge.engine import SimulationEngine
from converge.results import SimulationResults

__version__ = "1.0.0"
__all__ = [
    "Triangular", "PERT", "Uniform", "NormalTruncated",
    "Activity", "Network", "Predecessor",
    "NetworkError", "CycleError", "MissingPredecessorError", "InvalidRelationshipError",
    "parse_predecessor", "parse_predecessors", "format_predecessor", "format_predecessors",
    "SimulationEngine",
    "SimulationResults",
]
