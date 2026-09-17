# pencil-impact-crater

A custom initial-condition module for the [Pencil Code](https://github.com/pencil-code/pencil-code) that simulates a crater-forming impact.

`impact_crater.f90` deposits a calibrated amount of energy — split between internal heating and a downward velocity kick — into a buried region of the domain, representing the impact without simulating a resolved projectile. It's used to study the early excavation stage of impact cratering as a compressible hydrodynamics problem.

This code accompanies the paper "An energy-calibrated impact source for crater-excavation simulations in the Pencil Code" (R. Shukla, 2026).

## Usage

1. Copy `impact_crater.f90` into `src/initial_condition/` in a Pencil Code checkout (tested against revision `53438f895`).
2. Add to `src/Makefile.local`:
   ```
   INITIAL_CONDITION = initial_condition/impact_crater
   ```
3. Set the module's parameters in the `&initial_condition_pars` namelist in `start.in` (see the header comment in the source file for the full list).
