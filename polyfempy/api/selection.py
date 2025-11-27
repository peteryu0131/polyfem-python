"""Geometry selection utilities for boundary conditions.

This module provides Selection class for selecting boundary conditions
using geometric shapes (sphere, box, plane) without needing to know
specific sideset IDs from mesh files.
"""

import json
from typing import List, Dict, Any, Optional


class Selection:
    """Object used to select sidesets and bodies using geometric shapes.
    
    This class allows you to select boundary conditions by geometry rather
    than requiring explicit sideset IDs from mesh files. This is especially
    useful when mesh files don't have proper sideset markers.
    
    Example:
        >>> selection = Selection()
        >>> selection.select_sideset_with_sphere(id=1, center=[0, 0, 0], radius=1.0)
        >>> selection.select_sideset_with_box(id=2, box_min=[0, 0, 0], box_max=[1, 1, 1])
        >>> cfg = SimulationConfig(selection=selection)
    """
    
    def __init__(self):
        """Initialize an empty Selection object."""
        self.body_ids: List[Dict[str, Any]] = []
        self.boundary_sidesets: List[Dict[str, Any]] = []
    
    # Body selection methods
    
    def select_body_with_sphere(self, id: int, center: List[float], radius: float):
        """Select a body using a sphere.
        
        Args:
            id: Body ID to assign.
            center: Center of the sphere [x, y, z] or [x, y].
            radius: Radius of the sphere.
        """
        self.body_ids.append({"id": id, "center": center, "radius": radius})
    
    def select_body_with_box(self, id: int, box_min: List[float], box_max: List[float]):
        """Select a body using an axis-aligned box.
        
        Args:
            id: Body ID to assign.
            box_min: Minimum corner [x_min, y_min, z_min] or [x_min, y_min].
            box_max: Maximum corner [x_max, y_max, z_max] or [x_max, y_max].
        """
        self.body_ids.append({"id": id, "box": [box_min, box_max]})
    
    def select_body_with_axis_plane(self, id: int, axis: int, position: float):
        """Select a body using an axis-aligned plane.
        
        Args:
            id: Body ID to assign.
            axis: Axis direction (1=x, 2=y, 3=z). Use negative to flip (e.g., -1 is negative x).
            position: Position along the axis.
        """
        self.body_ids.append({"id": id, "position": position, "axis": axis})
    
    def select_body_with_plane(self, id: int, normal: List[float], offset: float):
        """Select a body using a generic plane.
        
        Args:
            id: Body ID to assign.
            normal: Normal vector of the plane [nx, ny, nz] or [nx, ny].
            offset: Offset of the plane. The point on the plane is defined by normal*offset.
        """
        self.body_ids.append({"id": id, "normal": normal, "offset": offset})
    
    # Sideset selection methods
    
    def select_sideset_with_sphere(self, id: int, center: List[float], radius: float):
        """Select a boundary sideset using a sphere.
        
        Args:
            id: Sideset ID to assign.
            center: Center of the sphere [x, y, z] or [x, y].
            radius: Radius of the sphere.
        """
        self.boundary_sidesets.append({"id": id, "center": center, "radius": radius})
    
    def select_sideset_with_box(self, id: int, box_min: List[float], box_max: List[float]):
        """Select a boundary sideset using an axis-aligned box.
        
        Args:
            id: Sideset ID to assign.
            box_min: Minimum corner [x_min, y_min, z_min] or [x_min, y_min].
            box_max: Maximum corner [x_max, y_max, z_max] or [x_max, y_max].
        """
        self.boundary_sidesets.append({"id": id, "box": [box_min, box_max]})
    
    def select_sideset_with_axis_plane(self, id: int, axis: int, position: float):
        """Select a boundary sideset using an axis-aligned plane.
        
        Args:
            id: Sideset ID to assign.
            axis: Axis direction (1=x, 2=y, 3=z). Use negative to flip (e.g., -1 is negative x).
            position: Position along the axis.
        """
        self.boundary_sidesets.append({"id": id, "position": position, "axis": axis})
    
    def select_sideset_with_plane(self, id: int, normal: List[float], offset: float):
        """Select a boundary sideset using a generic plane.
        
        Args:
            id: Sideset ID to assign.
            normal: Normal vector of the plane [nx, ny, nz] or [nx, ny].
            offset: Offset of the plane. The point on the plane is defined by normal*offset.
        """
        self.boundary_sidesets.append({"id": id, "normal": normal, "offset": offset})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Selection to a dictionary.
        
        Returns:
            Dictionary with 'body_ids' and 'boundary_sidesets' keys.
        """
        return {
            "body_ids": self.body_ids,
            "boundary_sidesets": self.boundary_sidesets,
        }
    
    def to_json_str(self) -> str:
        """Convert Selection to JSON string.
        
        Returns:
            JSON string representation of the selection.
        """
        return json.dumps(self.to_dict(), sort_keys=True, indent=4)
    
    def __str__(self) -> str:
        """String representation (JSON format).
        
        This matches the Legacy API's __str__ method which returns
        a JSON string representation of the selection.
        """
        # Match Legacy API: convert __dict__ to JSON with sort_keys and indent=4
        tmp = dict((key, value) for (key, value) in self.__dict__.items())
        return json.dumps(tmp, sort_keys=True, indent=4)
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"Selection(body_ids={len(self.body_ids)}, sidesets={len(self.boundary_sidesets)})"

