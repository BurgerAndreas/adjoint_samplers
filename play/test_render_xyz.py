#!/usr/bin/env python3
"""Test script for render_xyz_to_png function."""

from pathlib import Path
from adjoint_samplers.utils.eval_utils import (
    render_xyz_to_png,
    build_xyz_from_positions,
)
import numpy as np

# Create test output directory
output_dir = Path("test_output")
output_dir.mkdir(exist_ok=True)

# Test 1: Simple water molecule (H2O)
print("Test 1: Rendering water molecule (H2O)...")
water_xyz = """3
water
O    0.000000    0.000000    0.000000
H    0.957200    0.000000    0.000000
H   -0.240000    0.926600    0.000000"""

png_bytes = render_xyz_to_png(water_xyz, width=600, height=600)
output_path = output_dir / "test_water.png"
with open(output_path, "wb") as f:
    f.write(png_bytes)
print(f"  Saved to {output_path.resolve()}")

# Test 2: Methane (CH4)
print("\nTest 2: Rendering methane (CH4)...")
methane_xyz = """5
methane
C    0.000000    0.000000    0.000000
H    1.089000    0.000000    0.000000
H   -0.363000    1.027000    0.000000
H   -0.363000   -0.513500    0.890000
H   -0.363000   -0.513500   -0.890000"""

png_bytes = render_xyz_to_png(methane_xyz, width=600, height=600)
output_path = output_dir / "test_methane.png"
with open(output_path, "wb") as f:
    f.write(png_bytes)
print(f"  Saved to {output_path.resolve()}")

# Test 3: Using build_xyz_from_positions helper
print("\nTest 3: Rendering from positions array...")
# Create a simple linear molecule (3 atoms)
positions = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ]
)
xyz_str = build_xyz_from_positions(positions, atom_type="C", center=True)
png_bytes = render_xyz_to_png(xyz_str, width=400, height=400)
output_path = output_dir / "test_from_positions.png"
with open(output_path, "wb") as f:
    f.write(png_bytes)
print(f"  Saved to {output_path.resolve()}")

# Test 4: Different sizes
print("\nTest 4: Testing different image sizes...")
for size in [300, 600, 900]:
    png_bytes = render_xyz_to_png(water_xyz, width=size, height=size)
    output_path = output_dir / f"test_size_{size}.png"
    with open(output_path, "wb") as f:
        f.write(png_bytes)
    print(f"  Saved {size}x{size} to {output_path.resolve()}")

print("\nAll tests completed!")
