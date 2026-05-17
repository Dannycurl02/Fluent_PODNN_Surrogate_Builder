"""
Project Manager Module
======================
Queries Fluent for available inputs/outputs. No UI -- returns data for GUI to display.
"""

import logging
import re

logger = logging.getLogger(__name__)


# Regex for the value/unit format Fluent stores named-expression definitions in.
# Matches "1.5 [m/s]", "300[K]", "  -2.3 [degC]  ", etc.
_DEFINITION_RE = re.compile(
    r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\[([^\]]+)\]\s*$"
)


def _parse_definition(definition):
    """Parse a named-expression definition string into (value, unit_str).

    Returns (None, None) if it doesn't match the simple "<number> [<unit>]"
    shape — i.e. the expression is a formula, not a scalar input."""
    if not isinstance(definition, str):
        return None, None
    m = _DEFINITION_RE.match(definition)
    if not m:
        return None, None
    try:
        return float(m.group(1)), m.group(2)
    except ValueError:
        return None, None


def get_available_inputs(solver):
    """
    Query Fluent for available boundary conditions and input parameters
    (Fluent named expressions flagged as input_parameter).

    Cell zones are deliberately excluded — they don't expose anything cfdtwin
    can vary as a DOE input.

    Returns
    -------
    list of dict
        Each dict has keys: name, type, category. Input-parameter entries
        additionally carry: unit (str, e.g. "m/s"), current_value (float),
        definition (the raw "<value> [<unit>]" string).
    """
    items = []

    # Boundary conditions
    try:
        boundary_conditions = solver.settings.setup.boundary_conditions
        for bc_type in dir(boundary_conditions):
            if bc_type.startswith('_') or bc_type in ['child_names', 'command_names']:
                continue
            bc_obj = getattr(boundary_conditions, bc_type)
            if hasattr(bc_obj, '__iter__') and not isinstance(bc_obj, str):
                try:
                    for name in bc_obj:
                        if name not in ['child_names', 'command_names']:
                            items.append({
                                'name': name,
                                'type': bc_type.replace('_', ' ').title(),
                                'category': 'Boundary Condition'
                            })
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Error loading boundary conditions: {e}")

    # Input parameters (Fluent named expressions with input_parameter=True).
    # These have a "<value> [<unit>]" definition we can parse and re-write at
    # DOE time, which is cleaner than navigating the BC settings tree.
    try:
        named_exprs = solver.settings.setup.named_expressions
        for expr_name in named_exprs:
            if expr_name in ['child_names', 'command_names']:
                continue
            try:
                state = named_exprs[expr_name].get_state()
            except Exception:
                continue
            if not state.get('input_parameter'):
                continue
            value, unit = _parse_definition(state.get('definition', ''))
            if value is None:
                # Skip formula-style inputs we can't safely re-write as scalars.
                logger.info(
                    f"Skipping named expression '{expr_name}': "
                    f"definition {state.get('definition')!r} is not a scalar value."
                )
                continue
            items.append({
                'name': expr_name,
                'type': 'Input Parameter',
                'category': 'Input Parameter',
                'unit': unit,
                'current_value': value,
                'definition': state.get('definition', ''),
            })
    except Exception as e:
        logger.warning(f"Error loading input parameters: {e}")

    return items


def get_available_outputs(solver):
    """
    Query Fluent for available surfaces, cell zones, and report definitions that can be outputs.

    Returns
    -------
    list of dict
        Each dict has keys: name, type, category
    """
    items = []

    # Surfaces (from boundary conditions)
    try:
        boundary_conditions = solver.settings.setup.boundary_conditions
        for bc_type in dir(boundary_conditions):
            if bc_type.startswith('_') or bc_type in ['child_names', 'command_names']:
                continue
            bc_obj = getattr(boundary_conditions, bc_type)
            if hasattr(bc_obj, '__iter__') and not isinstance(bc_obj, str):
                try:
                    for name in bc_obj:
                        if name not in ['child_names', 'command_names']:
                            items.append({
                                'name': name,
                                'type': bc_type.replace('_', ' ').title(),
                                'category': 'Surface'
                            })
                except Exception:
                    pass

        # Created surfaces (planes, iso-surfaces, etc.)
        try:
            if hasattr(solver, 'fields') and hasattr(solver.fields, 'field_data'):
                all_surface_names = solver.fields.field_data.surfaces.allowed_values()
                for surf_name in all_surface_names:
                    if not any(s['name'] == surf_name for s in items):
                        items.append({
                            'name': surf_name,
                            'type': 'Created Surface',
                            'category': 'Surface'
                        })
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Error loading surfaces: {e}")

    # Cell zones
    try:
        cell_zones_obj = solver.settings.setup.cell_zone_conditions
        for zone_type in dir(cell_zones_obj):
            if zone_type.startswith('_') or zone_type in ['child_names', 'command_names']:
                continue
            zone_obj = getattr(cell_zones_obj, zone_type)
            if hasattr(zone_obj, '__iter__') and not isinstance(zone_obj, str):
                try:
                    for name in zone_obj:
                        if name not in ['child_names', 'command_names']:
                            items.append({
                                'name': name,
                                'type': zone_type.replace('_', ' ').title(),
                                'category': 'Cell Zone'
                            })
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Error loading cell zones: {e}")

    # Report definitions
    try:
        report_defs_obj = solver.settings.solution.report_definitions
        report_types = ['surface', 'volume', 'flux', 'force', 'lift', 'drag',
                        'moment', 'expression', 'user_defined']
        for report_type in report_types:
            if hasattr(report_defs_obj, report_type):
                report_obj = getattr(report_defs_obj, report_type)
                if hasattr(report_obj, '__iter__') and not isinstance(report_obj, str):
                    try:
                        for name in report_obj:
                            if name not in ['child_names', 'command_names']:
                                items.append({
                                    'name': name,
                                    'type': report_type.replace('_', ' ').title(),
                                    'category': 'Report Definition'
                                })
                    except Exception:
                        pass
    except Exception as e:
        logger.warning(f"Error loading report definitions: {e}")

    return items
