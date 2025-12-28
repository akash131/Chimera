"""Physics modules for Chimera."""

from chimera.physics.heat import HeatEquation, StefanBoltzmann
from chimera.physics.elasticity import LinearElasticity, PlaneStress, PlaneStrain
from chimera.physics.fluid import NavierStokes, Stokes, Euler

__all__ = [
    "HeatEquation",
    "StefanBoltzmann",
    "LinearElasticity",
    "PlaneStress",
    "PlaneStrain",
    "NavierStokes",
    "Stokes",
    "Euler",
]
