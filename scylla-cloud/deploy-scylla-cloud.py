#!/usr/bin/env python3
"""
ScyllaDB Cloud cluster management tool with vector search support.
Supports creating, destroying, checking status, and retrieving connection info.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import requests


API_BASE_URL = "https://api.cloud.scylladb.com"
STATE_FILE = Path.home() / ".scylla-clusters.json"


class ScyllaCloudClient:
    """Client for ScyllaDB Cloud REST API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def create_cluster(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new ScyllaDB cluster with vector search support."""
        url = f"{API_BASE_URL}/cluster/v1/clusters"
        response = requests.post(url, json=config, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_cluster(self, cluster_id: str) -> Dict[str, Any]:
        """Get cluster details by ID."""
        url = f"{API_BASE_URL}/cluster/v1/clusters/{cluster_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def list_clusters(self) -> Dict[str, Any]:
        """List all clusters."""
        url = f"{API_BASE_URL}/cluster/v1/clusters"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def delete_cluster(self, cluster_id: str) -> None:
        """Delete a cluster by ID."""
        url = f"{API_BASE_URL}/cluster/v1/clusters/{cluster_id}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
    
    def get_connection_info(self, cluster_id: str) -> Dict[str, Any]:
        """Get connection information for a cluster."""
        cluster = self.get_cluster(cluster_id)
        return {
            "cluster_id": cluster_id,
            "name": cluster.get("name"),
            "status": cluster.get("status"),
            "datacenter": cluster.get("datacenters", [{}])[0] if cluster.get("datacenters") else {},
            "connection": cluster.get("connection", {})
        }


class StateManager:
    """Manages local state file for cluster tracking."""
    
    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file
        self._ensure_state_file()
    
    def _ensure_state_file(self):
        """Create state file if it doesn't exist."""
        if not self.state_file.exists():
            self.state_file.write_text(json.dumps({"clusters": {}}, indent=2))
    
    def load_state(self) -> Dict[str, Any]:
        """Load state from file."""
        return json.loads(self.state_file.read_text())
    
    def save_state(self, state: Dict[str, Any]):
        """Save state to file."""
        self.state_file.write_text(json.dumps(state, indent=2))
    
    def add_cluster(self, name: str, cluster_data: Dict[str, Any]):
        """Add cluster to state."""
        state = self.load_state()
        state["clusters"][name] = cluster_data
        self.save_state(state)
    
    def get_cluster(self, name: str) -> Optional[Dict[str, Any]]:
        """Get cluster from state by name."""
        state = self.load_state()
        return state["clusters"].get(name)
    
    def remove_cluster(self, name: str):
        """Remove cluster from state."""
        state = self.load_state()
        if name in state["clusters"]:
            del state["clusters"][name]
            self.save_state(state)
    
    def list_clusters(self) -> Dict[str, Any]:
        """List all clusters in state."""
        state = self.load_state()
        return state["clusters"]


def output_result(data: Any, format_type: str):
    """Output data in specified format."""
    if format_type == "json":
        print(json.dumps(data, indent=2))
    else:
        # Human-readable text format
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    print(f"{key}:")
                    print(f"  {json.dumps(value, indent=2)}")
                else:
                    print(f"{key}: {value}")
        else:
            print(data)


def cmd_create(args):
    """Create a new ScyllaDB cluster with vector search."""
    client = ScyllaCloudClient(args.api_key)
    state_mgr = StateManager()
    
    # Check if cluster name already exists
    if state_mgr.get_cluster(args.name):
        print(f"✗ Error: Cluster '{args.name}' already exists in local state", file=sys.stderr)
        sys.exit(1)
    
    # Build cluster configuration
    config = {
        "name": args.name,
        "cloudProvider": args.cloud_provider,
        "region": args.region,
        "nodeCount": args.node_count,
        "nodeType": args.node_type,
    }
    
    # Add vector search configuration if enabled
    if args.enable_vector_search:
        config["vectorSearch"] = {
            "enabled": True,
            "nodeCount": args.vector_node_count,
            "nodeType": args.vector_node_type
        }
    
    # Add optional parameters
    if args.scylla_version:
        config["scyllaVersion"] = args.scylla_version
    if args.cidr_block:
        config["cidrBlock"] = args.cidr_block
    if args.enable_dns:
        config["enableDns"] = args.enable_dns
    if args.enable_vpc_peering:
        config["enableVpcPeering"] = args.enable_vpc_peering
    
    try:
        print(f"Creating cluster '{args.name}'...")
        result = client.create_cluster(config)
        
        # Save to state
        cluster_data = {
            "cluster_id": result.get("id"),
            "name": args.name,
            "cloud_provider": args.cloud_provider,
            "region": args.region,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": config
        }
        state_mgr.add_cluster(args.name, cluster_data)
        
        print(f"✓ Cluster creation initiated")
        print(f"Cluster ID: {result.get('id')}")
        print(f"Status: {result.get('status')}")
        print(f"\nState saved to {STATE_FILE}")
        
        output_result(result, args.format)
        
    except requests.exceptions.HTTPError as e:
        print(f"✗ API Error: {e}", file=sys.stderr)
        if e.response is not None:
            print(f"Response: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_destroy(args):
    """Destroy a ScyllaDB cluster."""
    client = ScyllaCloudClient(args.api_key)
    state_mgr = StateManager()
    
    # Get cluster from state
    cluster_data = state_mgr.get_cluster(args.name)
    if not cluster_data:
        print(f"✗ Error: Cluster '{args.name}' not found in local state", file=sys.stderr)
        sys.exit(1)
    
    cluster_id = cluster_data.get("cluster_id")
    if not cluster_id:
        print(f"✗ Error: No cluster ID found for '{args.name}'", file=sys.stderr)
        sys.exit(1)
    
    try:
        if not args.force:
            confirm = input(f"Are you sure you want to destroy cluster '{args.name}' ({cluster_id})? [y/N]: ")
            if confirm.lower() != 'y':
                print("Cancelled.")
                return
        
        print(f"Destroying cluster '{args.name}' ({cluster_id})...")
        client.delete_cluster(cluster_id)
        
        # Remove from state
        state_mgr.remove_cluster(args.name)
        
        print(f"✓ Cluster deletion initiated")
        print(f"Removed from state: {STATE_FILE}")
        
    except requests.exceptions.HTTPError as e:
        print(f"✗ API Error: {e}", file=sys.stderr)
        if e.response is not None:
            print(f"Response: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_status(args):
    """Get status of a ScyllaDB cluster."""
    client = ScyllaCloudClient(args.api_key)
    state_mgr = StateManager()
    
    # Get cluster from state
    cluster_data = state_mgr.get_cluster(args.name)
    if not cluster_data:
        print(f"✗ Error: Cluster '{args.name}' not found in local state", file=sys.stderr)
        sys.exit(1)
    
    cluster_id = cluster_data.get("cluster_id")
    if not cluster_id:
        print(f"✗ Error: No cluster ID found for '{args.name}'", file=sys.stderr)
        sys.exit(1)
    
    try:
        result = client.get_cluster(cluster_id)
        
        if args.format == "text":
            print(f"Cluster: {args.name}")
            print(f"ID: {cluster_id}")
            print(f"Status: {result.get('status')}")
            print(f"Cloud Provider: {result.get('cloudProvider')}")
            print(f"Region: {result.get('region')}")
            print(f"Node Count: {result.get('nodeCount')}")
            if result.get('vectorSearch'):
                print(f"Vector Search: Enabled")
                print(f"  Vector Node Count: {result['vectorSearch'].get('nodeCount')}")
        
        output_result(result, args.format)
        
    except requests.exceptions.HTTPError as e:
        print(f"✗ API Error: {e}", file=sys.stderr)
        if e.response is not None:
            print(f"Response: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_info(args):
    """Get connection information for a ScyllaDB cluster."""
    client = ScyllaCloudClient(args.api_key)
    state_mgr = StateManager()
    
    # Get cluster from state
    cluster_data = state_mgr.get_cluster(args.name)
    if not cluster_data:
        print(f"✗ Error: Cluster '{args.name}' not found in local state", file=sys.stderr)
        sys.exit(1)
    
    cluster_id = cluster_data.get("cluster_id")
    if not cluster_id:
        print(f"✗ Error: No cluster ID found for '{args.name}'", file=sys.stderr)
        sys.exit(1)
    
    try:
        result = client.get_connection_info(cluster_id)
        
        if args.format == "text":
            print(f"Connection Information for '{args.name}':")
            print(f"Cluster ID: {cluster_id}")
            print(f"Status: {result.get('status')}")
            
            connection = result.get("connection", {})
            if connection:
                print(f"\nConnection Details:")
                for key, value in connection.items():
                    print(f"  {key}: {value}")
        
        output_result(result, args.format)
        
    except requests.exceptions.HTTPError as e:
        print(f"✗ API Error: {e}", file=sys.stderr)
        if e.response is not None:
            print(f"Response: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_list(args):
    """List all clusters in local state."""
    state_mgr = StateManager()
    clusters = state_mgr.list_clusters()
    
    if args.format == "json":
        output_result(clusters, "json")
    else:
        if not clusters:
            print("No clusters found in local state.")
        else:
            print(f"Clusters in {STATE_FILE}:")
            for name, data in clusters.items():
                print(f"\n  {name}:")
                print(f"    Cluster ID: {data.get('cluster_id')}")
                print(f"    Cloud: {data.get('cloud_provider')}")
                print(f"    Region: {data.get('region')}")
                print(f"    Created: {data.get('created_at')}")


def main():
    parser = argparse.ArgumentParser(
        description="ScyllaDB Cloud cluster management with vector search support"
    )
    
    # Global arguments
    parser.add_argument(
        "--api-key",
        type=str,
        help="ScyllaDB Cloud API key (or set SCYLLA_CLOUD_API_KEY env var)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new cluster")
    create_parser.add_argument("--name", type=str, required=True, help="Cluster name")
    create_parser.add_argument(
        "--cloud-provider",
        type=str,
        choices=["AWS", "GCP"],
        default="AWS",
        help="Cloud provider (default: AWS)"
    )
    create_parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="Cloud region (default: us-east-1)"
    )
    create_parser.add_argument(
        "--node-count",
        type=int,
        default=3,
        help="Number of ScyllaDB nodes (default: 3, minimum: 3)"
    )
    create_parser.add_argument(
        "--node-type",
        type=str,
        default="i4i.large",
        help="Instance type for ScyllaDB nodes (default: i4i.large)"
    )
    create_parser.add_argument(
        "--enable-vector-search",
        action="store_true",
        help="Enable vector search capability"
    )
    create_parser.add_argument(
        "--vector-node-count",
        type=int,
        default=3,
        help="Number of vector search nodes (default: 3)"
    )
    create_parser.add_argument(
        "--vector-node-type",
        type=str,
        default="i4i.large",
        help="Instance type for vector search nodes (default: i4i.large)"
    )
    create_parser.add_argument(
        "--scylla-version",
        type=str,
        help="ScyllaDB version (optional)"
    )
    create_parser.add_argument(
        "--cidr-block",
        type=str,
        help="CIDR block for VPC (optional)"
    )
    create_parser.add_argument(
        "--enable-dns",
        action="store_true",
        help="Enable DNS (optional)"
    )
    create_parser.add_argument(
        "--enable-vpc-peering",
        action="store_true",
        help="Enable VPC peering (optional)"
    )
    
    # Destroy command
    destroy_parser = subparsers.add_parser("destroy", help="Destroy a cluster")
    destroy_parser.add_argument("--name", type=str, required=True, help="Cluster name")
    destroy_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get cluster status")
    status_parser.add_argument("--name", type=str, required=True, help="Cluster name")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get connection information")
    info_parser.add_argument("--name", type=str, required=True, help="Cluster name")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all clusters in local state")
    
    args = parser.parse_args()
    
    # Validate command
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Get API key
    api_key = args.api_key or os.getenv("SCYLLA_CLOUD_API_KEY")
    if not api_key and args.command != "list":
        print("✗ Error: API key required. Use --api-key or set SCYLLA_CLOUD_API_KEY", file=sys.stderr)
        sys.exit(1)
    
    args.api_key = api_key
    
    # Route to command handler
    if args.command == "create":
        if args.node_count < 3:
            print("✗ Error: Minimum node count is 3", file=sys.stderr)
            sys.exit(1)
        cmd_create(args)
    elif args.command == "destroy":
        cmd_destroy(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()
