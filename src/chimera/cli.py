"""
Command-line interface for Chimera.
"""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chimera - Hybrid Computational Physics Platform"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Version
    parser.add_argument('--version', action='version', version='chimera 0.1.0')

    # Solve command
    solve_parser = subparsers.add_parser('solve', help='Solve a physics problem')
    solve_parser.add_argument('input', help='Input file (JSON or YAML)')
    solve_parser.add_argument('-o', '--output', help='Output file')
    solve_parser.add_argument('-v', '--verbose', action='store_true')

    # Mesh command
    mesh_parser = subparsers.add_parser('mesh', help='Generate or convert meshes')
    mesh_parser.add_argument('action', choices=['generate', 'convert', 'info'])
    mesh_parser.add_argument('input', nargs='?', help='Input file')
    mesh_parser.add_argument('-o', '--output', help='Output file')

    # Visualize command
    vis_parser = subparsers.add_parser('viz', help='Visualize results')
    vis_parser.add_argument('input', help='Result file to visualize')
    vis_parser.add_argument('-f', '--field', default='u', help='Field to plot')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == 'solve':
        return cmd_solve(args)
    elif args.command == 'mesh':
        return cmd_mesh(args)
    elif args.command == 'viz':
        return cmd_viz(args)

    return 0


def cmd_solve(args):
    """Run solver from config file."""
    print(f"Solving: {args.input}")
    # Would load config and run solver
    return 0


def cmd_mesh(args):
    """Mesh operations."""
    if args.action == 'info':
        from chimera.mesh import Mesh
        mesh = Mesh.load(args.input)
        print(mesh.quality_metrics())
    return 0


def cmd_viz(args):
    """Visualize results."""
    from chimera.utils.io import load_results
    from chimera.utils.visualization import plot_field

    data = load_results(args.input)
    if args.field in data['fields']:
        values = data['fields'][args.field]['values']
        points = data['fields'][args.field]['points']
        plot_field(values, title=f"Field: {args.field}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
