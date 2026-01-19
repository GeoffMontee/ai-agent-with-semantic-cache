#!/usr/bin/env python3
"""
ScyllaDB Cloud cluster management tool with vector search support.
Supports creating, destroying, checking status, and retrieving connection info.
"""

import argparse
import ipaddress
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
    
    def __init__(self, api_key: str, debug: bool = False):
        self.api_key = api_key
        self.debug = debug
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def create_cluster(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new ScyllaDB cluster with vector search support."""
        account_id = config.get("accountId")
        url = f"{API_BASE_URL}/account/{account_id}/cluster"
        
        if self.debug:
            print(f"\n=== DEBUG: POST {url} ===")
            print(f"Headers: {json.dumps({k: v for k, v in self.headers.items() if k != 'Authorization'}, indent=2)}")
            print(f"Request Body: {json.dumps(config, indent=2)}")
        
        response = requests.post(url, json=config, headers=self.headers)
        
        if self.debug:
            print(f"\nResponse Status: {response.status_code}")
            print(f"Response Headers: {json.dumps(dict(response.headers), indent=2)}")
            try:
                print(f"Response Body: {json.dumps(response.json(), indent=2)}")
            except:
                print(f"Response Body: {response.text}")
            print("=== END DEBUG ===")
        
        if not response.ok:
            error_msg = f"HTTP {response.status_code} Error"
            try:
                error_body = response.json()
                error_msg += f"\n{json.dumps(error_body, indent=2)}"
            except:
                error_msg += f"\n{response.text}"
            raise requests.exceptions.HTTPError(error_msg, response=response)
        
        # Check for error in response body even with 200 status
        result = response.json()
        if isinstance(result, dict) and "error" in result:
            error_msg = f"API returned error: {result.get('error')}"
            if "message" in result:
                error_msg += f" - {result.get('message')}"
            raise requests.exceptions.HTTPError(error_msg, response=response)
        
        return result
    
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
    
    def list_clusters_by_account(self, account_id: str) -> Dict[str, Any]:
        """List all clusters for a specific account."""
        url = f"{API_BASE_URL}/account/{account_id}/clusters"
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
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get default account information."""
        url = f"{API_BASE_URL}/account/default"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_cloud_providers(self) -> Dict[str, Any]:
        """Get list of cloud providers."""
        url = f"{API_BASE_URL}/deployment/cloud-providers"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_cloud_provider_regions(self, cloud_provider_id: int) -> Dict[str, Any]:
        """Get list of regions for a cloud provider."""
        url = f"{API_BASE_URL}/deployment/cloud-provider/{cloud_provider_id}/regions"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_cloud_provider_region(self, cloud_provider_id: int, region_id: int, target: Optional[str] = None) -> Dict[str, Any]:
        """Get region details including instance types.
        
        Args:
            cloud_provider_id: Cloud provider ID
            region_id: Region ID
            target: Optional target parameter (e.g., 'VECTOR_SEARCH' for vector search instance types)
        """
        url = f"{API_BASE_URL}/deployment/cloud-provider/{cloud_provider_id}/region/{region_id}"
        if target:
            url += f"?target={target}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def lookup_cloud_provider_id(self, provider_name: str) -> int:
        """Look up cloud provider ID by name (AWS or GCP)."""
        response = self.get_cloud_providers()
        # Handle nested response structure
        if isinstance(response, dict):
            # Extract from nested structure: {"data": {"cloudProviders": [...]}}
            if 'data' in response:
                data = response['data']
                providers = data.get('cloudProviders', data.get('providers', []))
            else:
                providers = response.get('cloudProviders') or response.get('providers') or []
        else:
            providers = response
        
        for provider in providers:
            if isinstance(provider, dict) and provider.get('name', '').upper() == provider_name.upper():
                return provider['id']
        raise ValueError(f"Cloud provider '{provider_name}' not found")
    
    def lookup_region_id(self, cloud_provider_id: int, region_name: str) -> int:
        """Look up region ID by name."""
        response = self.get_cloud_provider_regions(cloud_provider_id)
        # Handle nested response structure
        if isinstance(response, dict):
            # Extract from nested structure: {"data": {"regions": [...]}}
            if 'data' in response:
                data = response['data']
                regions = data.get('regions') or data.get('cloudProviderRegions') or []
            else:
                regions = response.get('regions') or response.get('cloudProviderRegions') or []
        else:
            regions = response
        
        for region in regions:
            if isinstance(region, dict) and region.get('name') == region_name:
                return region['id']
        raise ValueError(f"Region '{region_name}' not found for cloud provider ID {cloud_provider_id}")
    
    def lookup_instance_type_id(self, cloud_provider_id: int, region_id: int, instance_type_name: str, target: Optional[str] = None) -> int:
        """Look up instance type ID by name.
        
        Args:
            cloud_provider_id: Cloud provider ID
            region_id: Region ID
            instance_type_name: Instance type name to look up
            target: Optional target parameter (e.g., 'VECTOR_SEARCH' for vector search instance types)
        """
        response = self.get_cloud_provider_region(cloud_provider_id, region_id, target)
        # Handle nested response structure
        if isinstance(response, dict):
            # Extract from nested structure if present
            if 'data' in response:
                region_details = response['data']
            else:
                region_details = response
        else:
            region_details = {}
        
        # API returns 'instances' not 'instanceTypes', and uses 'externalId' for the name
        instance_types = region_details.get('instances', region_details.get('instanceTypes', []))
        for instance_type in instance_types:
            if isinstance(instance_type, dict):
                # Check both externalId and name fields
                if instance_type.get('externalId') == instance_type_name or instance_type.get('name') == instance_type_name:
                    return instance_type['id']
        raise ValueError(f"Instance type '{instance_type_name}' not found in region ID {region_id}")


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


def validate_cidr_block(cidr: str, field_name: str):
    """Validate that CIDR block is /16 or larger (smaller prefix length)."""
    try:
        network = ipaddress.ip_network(cidr, strict=False)
        if network.prefixlen < 16:
            raise ValueError(
                f"{field_name} '{cidr}' has prefix length /{network.prefixlen}. "
                f"Must be /16 or larger (e.g., /16, /17, /24). "
                f"Larger prefix numbers = smaller networks."
            )
    except ValueError as e:
        if "prefix length" in str(e):
            raise
        raise ValueError(f"{field_name} '{cidr}' is not a valid CIDR block: {e}")


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
    client = ScyllaCloudClient(args.api_key, debug=args.debug)
    state_mgr = StateManager()
    
    # Check if cluster name already exists
    if state_mgr.get_cluster(args.name):
        print(f"✗ Error: Cluster '{args.name}' already exists in local state", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Look up IDs for cloud provider, region, and instance types
        print(f"Looking up cloud provider ID for '{args.cloud_provider}'...")
        if args.debug:
            providers_response = client.get_cloud_providers()
            print(f"[DEBUG] Cloud providers response: {json.dumps(providers_response, indent=2)}")
        cloud_provider_id = client.lookup_cloud_provider_id(args.cloud_provider)
        print(f"  Found cloud provider ID: {cloud_provider_id}")
        
        print(f"Looking up region ID for '{args.region}'...")
        if args.debug:
            regions_response = client.get_cloud_provider_regions(cloud_provider_id)
            print(f"[DEBUG] Regions response: {json.dumps(regions_response, indent=2)}")
        region_id = client.lookup_region_id(cloud_provider_id, args.region)
        print(f"  Found region ID: {region_id}")
        
        print(f"Looking up instance type ID for '{args.node_type}'...")
        if args.debug:
            instance_response = client.get_cloud_provider_region(cloud_provider_id, region_id)
            print(f"[DEBUG] Instance types response: {json.dumps(instance_response, indent=2)}")
        instance_id = client.lookup_instance_type_id(cloud_provider_id, region_id, args.node_type)
        print(f"  Found instance type ID: {instance_id}")
        
        # Look up vector instance type ID if vector search is enabled
        vector_instance_id = None
        if args.enable_vector_search:
            print(f"Looking up vector instance type ID for '{args.vector_node_type}'...")
            vector_instance_id = client.lookup_instance_type_id(cloud_provider_id, region_id, args.vector_node_type, target="VECTOR_SEARCH")
            print(f"  Found vector instance type ID: {vector_instance_id}")
    
    except ValueError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate CIDR blocks
    try:
        validate_cidr_block(args.cidr_block, "--cidr-block")
        for ip in args.allowed_ips:
            validate_cidr_block(ip, "--allowed-ips")
    except ValueError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Build cluster configuration
    free_tier = args.free_tier or os.getenv("SCYLLA_CLOUD_FREE_TIER", "false").lower() == "true"
    broadcast_type = args.broadcast_type
    user_api = args.user_api or os.getenv("SCYLLA_CLOUD_USER_API", "CQL")
    
    config = {
        "clusterName": args.name,
        "accountId": args.account_id,
        "cloudProviderId": cloud_provider_id,
        "regionId": region_id,
        "instanceId": instance_id,
        "numberOfNodes": args.node_count,
        "broadcastType": broadcast_type,
        "replicationFactor": args.replication_factor,
        "tablets": "false" if args.disable_tablets else "enforced",
        "allowedIPs": args.allowed_ips,
        "scyllaVersion": args.scylla_version or os.getenv("SCYLLA_VERSION", "2025.4.0"),
        "accountCredentialId": args.account_credential_id if args.account_credential_id is not None else int(os.getenv("SCYLLA_CLOUD_ACCOUNT_CREDENTIAL_ID", "3")),
        "freeTier": free_tier,
        "promProxy": args.prometheus_proxy or os.getenv("SCYLLA_CLOUD_PROMETHEUS_PROXY", "false").lower() == "true",
        "userApiInterface": user_api,
        "enableDnsAssociation": args.enable_dns_association or os.getenv("SCYLLA_CLOUD_ENABLE_DNS_ASSOCIATION", "true").lower() == "true",
        "provisioning": args.provisioning or os.getenv("SCYLLA_CLOUD_PROVISIONING", "dedicated-vm"),
    }
    
    # Add cidrBlock only for PRIVATE broadcast type
    if broadcast_type == "PRIVATE":
        config["cidrBlock"] = args.cidr_block
    
    # Add alternatorWriteIsolation only for ALTERNATOR user API
    if user_api == "ALTERNATOR":
        config["alternatorWriteIsolation"] = args.alternator_write_isolation or os.getenv("SCYLLA_CLOUD_ALTERNATOR_WRITE_ISOLATION", "only_rmw_uses_lwt")
    
    # Add pu and expiration only for free tier clusters
    if free_tier:
        config["pu"] = args.pu if args.pu is not None else int(os.getenv("SCYLLA_CLOUD_PU", "1"))
        config["expiration"] = args.expiration or os.getenv("SCYLLA_CLOUD_EXPIRATION", "0")
    
    # Add vector search configuration if enabled
    if args.enable_vector_search:
        config["vectorSearch"] = {
            "defaultNodes": args.vector_node_count,
            "defaultInstanceTypeId": vector_instance_id,
            "singleRack": args.vector_single_rack or os.getenv("SCYLLA_CLOUD_VECTOR_SINGLE_RACK", "false").lower() == "true"
        }
    
    # Add optional parameters
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
            "account_id": args.account_id,
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
        print(f"\n✗ API Error: {e}", file=sys.stderr)
        if e.response is not None:
            print(f"\nHTTP Status: {e.response.status_code}", file=sys.stderr)
            print(f"URL: {e.response.url}", file=sys.stderr)
            try:
                error_body = e.response.json()
                print(f"\nError Details:", file=sys.stderr)
                print(json.dumps(error_body, indent=2), file=sys.stderr)
            except:
                print(f"\nResponse Body:", file=sys.stderr)
                print(e.response.text, file=sys.stderr)
        if not args.debug:
            print(f"\nTip: Run with --debug flag for full request/response details", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_destroy(args):
    """Destroy a ScyllaDB cluster."""
    client = ScyllaCloudClient(args.api_key, debug=args.debug)
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
    client = ScyllaCloudClient(args.api_key, debug=args.debug)
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
    client = ScyllaCloudClient(args.api_key, debug=args.debug)
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


def cmd_list_clusters(args):
    """List all clusters for an account from ScyllaDB Cloud API."""
    client = ScyllaCloudClient(args.api_key, debug=args.debug)
    
    try:
        result = client.list_clusters_by_account(args.account_id)
        
        if args.format == "text":
            print(f"Clusters for Account ID: {args.account_id}")
            print("-" * 80)
        
        output_result(result, args.format)
        
        if args.format == "text":
            print("-" * 80)
        
    except requests.exceptions.HTTPError as e:
        print(f"✗ API Error: {e}", file=sys.stderr)
        if e.response is not None:
            print(f"Response: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_get_account_info(args):
    """Get ScyllaDB Cloud account information."""
    client = ScyllaCloudClient(args.api_key, debug=args.debug)
    
    try:
        result = client.get_account_info()
        
        if args.format == "text":
            print("ScyllaDB Cloud Account Information:")
            print("-" * 80)
        
        output_result(result, args.format)
        
        if args.format == "text":
            print("-" * 80)
        
    except requests.exceptions.HTTPError as e:
        print(f"✗ API Error: {e}", file=sys.stderr)
        if e.response is not None:
            print(f"Response: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (shows full request/response details)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new cluster")
    create_parser.add_argument("--name", type=str, required=True, help="Cluster name")
    create_parser.add_argument(
        "--account-id",
        type=str,
        help="ScyllaDB Cloud account ID (required, or set SCYLLA_CLOUD_ACCOUNT_ID env var)"
    )
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
        "--disable-vector-search",
        action="store_false",
        dest="enable_vector_search",
        help="Disable vector search capability (vector search is enabled by default)"
    )
    create_parser.add_argument(
        "--vector-node-count",
        type=int,
        default=1,
        help="Number of vector search nodes (default: 1)"
    )
    create_parser.add_argument(
        "--vector-node-type",
        type=str,
        default="i4i.large",
        help="Instance type for vector search nodes (default: i4i.large)"
    )
    create_parser.add_argument(
        "--vector-single-rack",
        action="store_true",
        help="Enable single rack for vector search (default: false, or set SCYLLA_CLOUD_VECTOR_SINGLE_RACK env var)"
    )
    create_parser.add_argument(
        "--broadcast-type",
        type=str,
        choices=["PUBLIC", "PRIVATE"],
        default="PUBLIC",
        help="Broadcast type (default: PUBLIC)"
    )
    create_parser.add_argument(
        "--cidr-block",
        type=str,
        default="192.168.1.0/24",
        help="CIDR block for VPC (default: 192.168.1.0/24)"
    )
    create_parser.add_argument(
        "--allowed-ips",
        type=str,
        nargs="*",
        default=["0.0.0.0/0"],
        help="Allowed IP addresses (default: 0.0.0.0/0, space-separated list)"
    )
    create_parser.add_argument(
        "--replication-factor",
        type=int,
        default=3,
        help="Replication factor (default: 3)"
    )
    create_parser.add_argument(
        "--disable-tablets",
        action="store_true",
        help="Disable tablets (enabled by default)"
    )
    create_parser.add_argument(
        "--scylla-version",
        type=str,
        default="2025.4.0",
        help="ScyllaDB version (default: 2025.4.0, or set SCYLLA_VERSION env var)"
    )
    create_parser.add_argument(
        "--account-credential-id",
        type=int,
        default=3,
        help="Account credential ID (default: 3, or set SCYLLA_CLOUD_ACCOUNT_CREDENTIAL_ID env var)"
    )
    create_parser.add_argument(
        "--alternator-write-isolation",
        type=str,
        default="only_rmw_uses_lwt",
        help="Alternator write isolation (default: only_rmw_uses_lwt, or set SCYLLA_CLOUD_ALTERNATOR_WRITE_ISOLATION env var)"
    )
    create_parser.add_argument(
        "--free-tier",
        action="store_true",
        help="Enable free tier (default: false, or set SCYLLA_CLOUD_FREE_TIER env var)"
    )
    create_parser.add_argument(
        "--prometheus-proxy",
        action="store_true",
        help="Enable Prometheus proxy (default: false, or set SCYLLA_CLOUD_PROMETHEUS_PROXY env var)"
    )
    create_parser.add_argument(
        "--user-api",
        type=str,
        default="CQL",
        help="User API interface (default: CQL, or set SCYLLA_CLOUD_USER_API env var)"
    )
    create_parser.add_argument(
        "--enable-dns-association",
        action="store_true",
        help="Enable DNS association (default: true, or set SCYLLA_CLOUD_ENABLE_DNS_ASSOCIATION env var)"
    )
    create_parser.add_argument(
        "--provisioning",
        type=str,
        default="dedicated-vm",
        help="Provisioning type (default: dedicated-vm, or set SCYLLA_CLOUD_PROVISIONING env var)"
    )
    create_parser.add_argument(
        "--pu",
        type=int,
        default=1,
        help="Processing units (default: 1, or set SCYLLA_CLOUD_PU env var)"
    )
    create_parser.add_argument(
        "--expiration",
        type=str,
        default="0",
        help="Expiration (default: 0, or set SCYLLA_CLOUD_EXPIRATION env var)"
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
    
    # List clusters command
    list_clusters_parser = subparsers.add_parser("list-clusters", help="List all clusters for an account from ScyllaDB Cloud API")
    list_clusters_parser.add_argument(
        "--account-id",
        type=str,
        help="ScyllaDB Cloud account ID (required, or set SCYLLA_CLOUD_ACCOUNT_ID env var)"
    )
    
    # Get account info command
    account_parser = subparsers.add_parser("get-account-info", help="Get ScyllaDB Cloud account information")
    
    args = parser.parse_args()
    
    # Validate command
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Get API key (not required for 'list' command)
    api_key = args.api_key or os.getenv("SCYLLA_CLOUD_API_KEY")
    if not api_key and args.command not in ["list"]:
        print("✗ Error: API key required. Use --api-key or set SCYLLA_CLOUD_API_KEY", file=sys.stderr)
        sys.exit(1)
    
    args.api_key = api_key
    
    # Route to command handler
    if args.command == "create":
        if args.node_count < 3:
            print("✗ Error: Minimum node count is 3", file=sys.stderr)
            sys.exit(1)
        # Validate account ID
        if not args.account_id:
            args.account_id = os.getenv("SCYLLA_CLOUD_ACCOUNT_ID")
        if not args.account_id:
            print("✗ Error: Account ID is required. Use --account-id or set SCYLLA_CLOUD_ACCOUNT_ID", file=sys.stderr)
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
    elif args.command == "list-clusters":
        # Validate account ID
        if not args.account_id:
            args.account_id = os.getenv("SCYLLA_CLOUD_ACCOUNT_ID")
        if not args.account_id:
            print("✗ Error: Account ID is required. Use --account-id or set SCYLLA_CLOUD_ACCOUNT_ID", file=sys.stderr)
            sys.exit(1)
        cmd_list_clusters(args)
    elif args.command == "get-account-info":
        cmd_get_account_info(args)


if __name__ == "__main__":
    main()
