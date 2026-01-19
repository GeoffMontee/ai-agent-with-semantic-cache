#!/usr/bin/env python3
"""
PostgreSQL with pgvector Docker deployment tool.
Manages local PostgreSQL instances with pgvector extension using Docker Compose.
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import docker
except ImportError:
    print("Error: docker package not installed. Run: pip install docker", file=sys.stderr)
    sys.exit(1)


SCRIPT_DIR = Path(__file__).parent.absolute()
COMPOSE_FILE = SCRIPT_DIR / "docker-compose.yml"


class DockerManager:
    """Manages Docker containers and operations."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        try:
            self.client = docker.from_env()
        except Exception as e:
            print(f"✗ Error: Could not connect to Docker. Is Docker running?", file=sys.stderr)
            print(f"  {e}", file=sys.stderr)
            sys.exit(1)
    
    def get_container(self, name: str) -> Optional[Any]:
        """Get container by name if it exists."""
        try:
            return self.client.containers.get(name)
        except docker.errors.NotFound:
            return None
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Error getting container: {e}")
            return None
    
    def is_port_in_use(self, port: int, exclude_container: Optional[str] = None) -> bool:
        """Check if a port is already in use by another container."""
        for container in self.client.containers.list():
            if exclude_container and container.name == exclude_container:
                continue
            for port_binding in container.attrs.get('NetworkSettings', {}).get('Ports', {}).values():
                if port_binding:
                    for binding in port_binding:
                        if binding and int(binding.get('HostPort', 0)) == port:
                            return True
        return False


def run_compose_command(args: list, env: Dict[str, str], debug: bool = False) -> subprocess.CompletedProcess:
    """Run a docker-compose command with the given environment."""
    cmd = ["docker-compose", "-f", str(COMPOSE_FILE)] + args
    
    if debug:
        print(f"\n=== DEBUG: Running command ===")
        print(f"Command: {' '.join(cmd)}")
        print(f"Environment: {json.dumps({k: v for k, v in env.items() if 'PASSWORD' not in k}, indent=2)}")
        print("=== END DEBUG ===\n")
    
    return subprocess.run(cmd, env={**os.environ, **env}, capture_output=True, text=True)


def create_override_file(config: Dict[str, str]) -> Path:
    """Create docker-compose.override.yml with custom configuration."""
    override_file = SCRIPT_DIR / "docker-compose.override.yml"
    
    # Only create override if there are custom ports or specific config
    # For now, we'll use environment variables passed to docker-compose
    # This keeps things simpler
    
    return override_file


def cmd_start(args):
    """Start PostgreSQL container with pgvector."""
    docker_mgr = DockerManager(debug=args.debug)
    
    # Check if port is already in use
    if docker_mgr.is_port_in_use(args.port, exclude_container=args.name):
        print(f"✗ Error: Port {args.port} is already in use by another container", file=sys.stderr)
        sys.exit(1)
    
    # Check if container already exists
    existing = docker_mgr.get_container(args.name)
    if existing:
        if existing.status == "running":
            print(f"✓ Container '{args.name}' is already running")
            return
        else:
            print(f"Starting existing container '{args.name}'...")
    else:
        print(f"Creating and starting container '{args.name}'...")
    
    # Build environment for docker-compose
    env = {
        "CONTAINER_NAME": args.name,
        "POSTGRES_VERSION": f"pg{args.postgres_version}",
        "POSTGRES_USER": args.user,
        "POSTGRES_PASSWORD": args.password,
        "POSTGRES_DB": args.database,
        "POSTGRES_PORT": str(args.port),
        "VOLUME_NAME": f"{args.name}-data",
    }
    
    # Start container
    result = run_compose_command(["up", "-d"], env, debug=args.debug)
    
    if result.returncode != 0:
        print(f"✗ Error starting container:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    
    print(f"✓ Container '{args.name}' started successfully")
    print(f"\nConnection details:")
    print(f"  Host: localhost")
    print(f"  Port: {args.port}")
    print(f"  User: {args.user}")
    print(f"  Database: {args.database}")
    print(f"  Password: {args.password}")
    print(f"\nConnection string:")
    print(f"  postgresql://{args.user}:{args.password}@localhost:{args.port}/{args.database}")
    
    # Wait for container to be healthy
    print(f"\nWaiting for PostgreSQL to be ready...")
    container = docker_mgr.get_container(args.name)
    if container:
        import time
        max_wait = 30
        waited = 0
        while waited < max_wait:
            container.reload()
            health = container.attrs.get('State', {}).get('Health', {}).get('Status')
            if health == 'healthy':
                print("✓ PostgreSQL is ready and pgvector extension is installed")
                break
            time.sleep(1)
            waited += 1
        else:
            print("⚠ PostgreSQL started but health check timed out. Check logs with: ./deploy-pgvector-docker.py logs")


def cmd_stop(args):
    """Stop PostgreSQL container (preserves data)."""
    docker_mgr = DockerManager(debug=args.debug)
    
    container = docker_mgr.get_container(args.name)
    if not container:
        print(f"✗ Error: Container '{args.name}' not found", file=sys.stderr)
        sys.exit(1)
    
    if container.status != "running":
        print(f"✓ Container '{args.name}' is already stopped")
        return
    
    print(f"Stopping container '{args.name}'...")
    
    env = {"CONTAINER_NAME": args.name}
    result = run_compose_command(["stop"], env, debug=args.debug)
    
    if result.returncode != 0:
        print(f"✗ Error stopping container:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    
    print(f"✓ Container '{args.name}' stopped (data preserved)")


def cmd_restart(args):
    """Restart PostgreSQL container."""
    docker_mgr = DockerManager(debug=args.debug)
    
    container = docker_mgr.get_container(args.name)
    if not container:
        print(f"✗ Error: Container '{args.name}' not found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Restarting container '{args.name}'...")
    
    env = {"CONTAINER_NAME": args.name}
    result = run_compose_command(["restart"], env, debug=args.debug)
    
    if result.returncode != 0:
        print(f"✗ Error restarting container:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    
    print(f"✓ Container '{args.name}' restarted")


def cmd_destroy(args):
    """Stop and remove container and optionally volumes."""
    docker_mgr = DockerManager(debug=args.debug)
    
    container = docker_mgr.get_container(args.name)
    if not container:
        print(f"✗ Error: Container '{args.name}' not found", file=sys.stderr)
        sys.exit(1)
    
    if not args.force:
        volume_msg = " and its data volume" if args.remove_volumes else ""
        confirm = input(f"Are you sure you want to destroy '{args.name}'{volume_msg}? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return
    
    print(f"Destroying container '{args.name}'...")
    
    env = {"CONTAINER_NAME": args.name, "VOLUME_NAME": f"{args.name}-data"}
    cmd = ["down"]
    if args.remove_volumes:
        cmd.append("-v")
    
    result = run_compose_command(cmd, env, debug=args.debug)
    
    if result.returncode != 0:
        print(f"✗ Error destroying container:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    
    msg = "✓ Container destroyed"
    if args.remove_volumes:
        msg += " (data volume removed)"
    else:
        msg += " (data volume preserved)"
    print(msg)


def cmd_status(args):
    """Check PostgreSQL container status."""
    docker_mgr = DockerManager(debug=args.debug)
    
    container = docker_mgr.get_container(args.name)
    if not container:
        print(f"Container '{args.name}': Not found")
        return
    
    container.reload()
    status = container.status
    health = container.attrs.get('State', {}).get('Health', {}).get('Status', 'N/A')
    
    print(f"Container '{args.name}':")
    print(f"  Status: {status}")
    print(f"  Health: {health}")
    
    if status == "running":
        ports = container.attrs.get('NetworkSettings', {}).get('Ports', {})
        for container_port, bindings in ports.items():
            if bindings:
                for binding in bindings:
                    print(f"  Port: {binding['HostIp']}:{binding['HostPort']} -> {container_port}")
    
    # Show volumes
    mounts = container.attrs.get('Mounts', [])
    for mount in mounts:
        if mount['Type'] == 'volume':
            print(f"  Volume: {mount['Name']} -> {mount['Destination']}")


def cmd_info(args):
    """Get connection information for PostgreSQL container."""
    docker_mgr = DockerManager(debug=args.debug)
    
    container = docker_mgr.get_container(args.name)
    if not container:
        print(f"✗ Error: Container '{args.name}' not found", file=sys.stderr)
        sys.exit(1)
    
    container.reload()
    
    if container.status != "running":
        print(f"⚠ Warning: Container '{args.name}' is not running (status: {container.status})")
    
    # Extract connection info from environment
    env = container.attrs.get('Config', {}).get('Env', [])
    env_dict = {}
    for item in env:
        if '=' in item:
            key, value = item.split('=', 1)
            env_dict[key] = value
    
    user = env_dict.get('POSTGRES_USER', 'postgres')
    password = env_dict.get('POSTGRES_PASSWORD', 'postgres')
    database = env_dict.get('POSTGRES_DB', 'postgres')
    
    # Get port from port bindings
    port = None
    ports = container.attrs.get('NetworkSettings', {}).get('Ports', {})
    for bindings in ports.values():
        if bindings:
            port = bindings[0]['HostPort']
            break
    
    if args.format == "json":
        info = {
            "container": args.name,
            "status": container.status,
            "host": "localhost",
            "port": int(port) if port else None,
            "user": user,
            "password": password,
            "database": database,
            "connection_string": f"postgresql://{user}:{password}@localhost:{port}/{database}" if port else None
        }
        print(json.dumps(info, indent=2))
    else:
        print(f"Connection Information for '{args.name}':")
        print(f"  Status: {container.status}")
        print(f"  Host: localhost")
        print(f"  Port: {port}")
        print(f"  User: {user}")
        print(f"  Database: {database}")
        print(f"  Password: {password}")
        if port:
            print(f"\nConnection string:")
            print(f"  postgresql://{user}:{password}@localhost:{port}/{database}")


def cmd_logs(args):
    """View container logs."""
    docker_mgr = DockerManager(debug=args.debug)
    
    container = docker_mgr.get_container(args.name)
    if not container:
        print(f"✗ Error: Container '{args.name}' not found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Logs for container '{args.name}':")
    print("-" * 80)
    
    # Get logs
    logs = container.logs(tail=args.tail, follow=args.follow, stream=args.follow)
    
    if args.follow:
        try:
            for line in logs:
                print(line.decode('utf-8'), end='')
        except KeyboardInterrupt:
            print("\n[Stopped following logs]")
    else:
        print(logs.decode('utf-8'))


def main():
    parser = argparse.ArgumentParser(
        description="PostgreSQL with pgvector Docker deployment tool"
    )
    
    # Global arguments
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start PostgreSQL container")
    start_parser.add_argument(
        "--name",
        type=str,
        default="pgvector-local",
        help="Container name (default: pgvector-local)"
    )
    start_parser.add_argument(
        "--postgres-version",
        type=int,
        default=18,
        choices=[12, 13, 14, 15, 16, 17, 18],
        help="PostgreSQL version (default: 18)"
    )
    start_parser.add_argument(
        "--user",
        type=str,
        default="postgres",
        help="PostgreSQL user (default: postgres)"
    )
    start_parser.add_argument(
        "--password",
        type=str,
        default="postgres",
        help="PostgreSQL password (default: postgres)"
    )
    start_parser.add_argument(
        "--database",
        type=str,
        default="postgres",
        help="PostgreSQL database name (default: postgres)"
    )
    start_parser.add_argument(
        "--port",
        type=int,
        default=5432,
        help="PostgreSQL port (default: 5432)"
    )
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop PostgreSQL container")
    stop_parser.add_argument(
        "--name",
        type=str,
        default="pgvector-local",
        help="Container name (default: pgvector-local)"
    )
    
    # Restart command
    restart_parser = subparsers.add_parser("restart", help="Restart PostgreSQL container")
    restart_parser.add_argument(
        "--name",
        type=str,
        default="pgvector-local",
        help="Container name (default: pgvector-local)"
    )
    
    # Destroy command
    destroy_parser = subparsers.add_parser("destroy", help="Destroy PostgreSQL container")
    destroy_parser.add_argument(
        "--name",
        type=str,
        default="pgvector-local",
        help="Container name (default: pgvector-local)"
    )
    destroy_parser.add_argument(
        "--remove-volumes",
        action="store_true",
        help="Also remove data volumes (deletes all data)"
    )
    destroy_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check container status")
    status_parser.add_argument(
        "--name",
        type=str,
        default="pgvector-local",
        help="Container name (default: pgvector-local)"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get connection information")
    info_parser.add_argument(
        "--name",
        type=str,
        default="pgvector-local",
        help="Container name (default: pgvector-local)"
    )
    
    # Logs command
    logs_parser = subparsers.add_parser("logs", help="View container logs")
    logs_parser.add_argument(
        "--name",
        type=str,
        default="pgvector-local",
        help="Container name (default: pgvector-local)"
    )
    logs_parser.add_argument(
        "--tail",
        type=int,
        default=100,
        help="Number of lines to show from end of logs (default: 100)"
    )
    logs_parser.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="Follow log output"
    )
    
    args = parser.parse_args()
    
    # Validate command
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to command handler
    if args.command == "start":
        cmd_start(args)
    elif args.command == "stop":
        cmd_stop(args)
    elif args.command == "restart":
        cmd_restart(args)
    elif args.command == "destroy":
        cmd_destroy(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "logs":
        cmd_logs(args)


if __name__ == "__main__":
    main()
