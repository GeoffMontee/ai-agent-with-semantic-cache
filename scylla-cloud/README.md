# ScyllaDB Cloud Deployment Tool

A command-line tool for managing ScyllaDB Cloud clusters with vector search support using the ScyllaDB Cloud REST API.

## Features

- 🚀 **Create Clusters**: Deploy ScyllaDB clusters with optional vector search nodes
- � **Smart ID Lookups**: Automatically translates cloud provider, region, and instance type names to API IDs
- 🗑️ **Destroy Clusters**: Clean up clusters when no longer needed
- 📊 **Status Monitoring**: Check cluster status and health
- 🔌 **Connection Info**: Retrieve connection details for client applications
- 💾 **State Management**: Track clusters locally for easy management
- 🎛️ **Flexible Output**: Human-readable text or JSON for scripting
- 🔧 **Debug Mode**: Full request/response logging for troubleshooting

## Requirements

- Python 3.8+
- ScyllaDB Cloud API key
- `requests` library

## Installation

Install the required Python package:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install requests
```

Make the script executable:

```bash
chmod +x deploy-scylla-cloud.py
```

## Configuration

### API Key

Set your ScyllaDB Cloud API key either via:

1. **Environment variable** (recommended):
```bash
export SCYLLA_CLOUD_API_KEY="your-api-key"
```

2. **Command-line argument**:
```bash
./deploy-scylla-cloud.py create --api-key "your-api-key" --name mycluster
```

### State File

The tool stores cluster information locally in `~/.scylla-clusters.json`. This file tracks:
- Cluster names and IDs
- Configuration details
- Creation timestamps

## Usage

### Create a Cluster

**Basic cluster (vector search enabled by default):**
```bash
./deploy-scylla-cloud.py create \
  --name mycluster \
  --account-id "your-account-id"
```

This creates a cluster with:
- 3 ScyllaDB nodes (i4i.large instances)
- 3 vector search nodes (i4i.large instances)
- AWS cloud provider in us-east-1 region
- PUBLIC broadcast type
- CIDR block: 192.168.1.0/24
- Allowed IPs: 0.0.0.0/0 (all IPs)
- Replication factor: 3
- Tablets enabled
- ScyllaDB version: 2025.4.1
- User API: CQL
- Alternator write isolation: only_rmw_uses_lwt

The tool automatically looks up the cloud provider ID, region ID, and instance type IDs from the ScyllaDB Cloud API.

**With custom vector search configuration:**
```bash
./deploy-scylla-cloud.py create \
  --name vectorcluster \
  --account-id "your-account-id" \
  --cloud-provider AWS \
  --region us-east-1 \
  --node-count 3 \
  --node-type i4i.large \
  --vector-node-count 3 \
  --vector-node-type i4i.large
```

**Full configuration example:**
```bash
./deploy-scylla-cloud.py create \
  --name production-cluster \
  --account-id "your-account-id" \
  --cloud-provider GCP \
  --region us-central1 \
  --node-count 5 \
  --node-type n2-highmem-8 \
  --vector-node-count 3 \
  --vector-node-type n2-highmem-4 \
  --broadcast-type PRIVATE \
  --cidr-block "10.0.0.0/16" \
  --allowed-ips "203.0.113.0/24" "198.51.100.0/24" \
  --replication-factor 3 \
  --scylla-version "2025.4.1" \
  --account-credential-id 0 \
  --alternator-write-isolation "only_rmw_uses_lwt" \
  --user-api "CQL" \
  --enable-dns \
  --enable-vpc-peering
```

### Check Cluster Status

```bash
./deploy-scylla-cloud.py status --name mycluster
```

### Get Connection Information

```bash
./deploy-scylla-cloud.py info --name mycluster
```

Output includes:
- Cluster ID and status
- Connection endpoints
- Credentials (if available)
- Port numbers

### List All Clusters

```bash
./deploy-scylla-cloud.py list
```

### Get Account Information

Retrieve your ScyllaDB Cloud account information, including the account ID needed for cluster creation:

```bash
./deploy-scylla-cloud.py get-account-info
```

This command is useful for finding your account ID when you need to create a new cluster. The account ID is required by the `create` command.

**JSON output:**
```bash
./deploy-scylla-cloud.py get-account-info --format json
```

### Destroy a Cluster

**With confirmation prompt:**
```bash
./deploy-scylla-cloud.py destroy --name mycluster
```

**Skip confirmation (use with caution):**
```bash
./deploy-scylla-cloud.py destroy --name mycluster --force
```

## Command Reference

### Global Options

| Option | Description | Default |
|--------|-------------|---------|
| `--api-key` | ScyllaDB Cloud API key | `SCYLLA_CLOUD_API_KEY` env var |
| `--format` | Output format: `text` or `json` | `text` |
| `--debug` | Enable debug output with full request/response details | `False` |

### Create Command

| Option | Description | Default |
|--------|-------------|---------|
| `--name` | Cluster name (required) | - |
| `--account-id` | ScyllaDB Cloud account ID (required) | `SCYLLA_CLOUD_ACCOUNT_ID` env var |
| `--cloud-provider` | Cloud provider: `AWS` or `GCP` | `AWS` |
| `--region` | Cloud region | `us-east-1` |
| `--node-count` | Number of ScyllaDB nodes (min: 3) | `3` |
| `--node-type` | Instance type for ScyllaDB nodes | `i4i.large` |
| `--disable-vector-search` | Disable vector search (enabled by default) | `False` |
| `--vector-node-count` | Number of vector search nodes | `3` |
| `--vector-node-type` | Instance type for vector nodes | `i4i.large` |
| `--broadcast-type` | Broadcast type: `PUBLIC` or `PRIVATE` | `PUBLIC` |
| `--cidr-block` | VPC CIDR block | `192.168.1.0/24` |
| `--allowed-ips` | Allowed IP addresses (space-separated) | `0.0.0.0/0` |
| `--replication-factor` | Replication factor | `3` |
| `--disable-tablets` | Disable tablets (enabled by default) | `False` |
| `--scylla-version` | ScyllaDB version | `2025.4.1` |
| `--account-credential-id` | Account credential ID | `0` |
| `--alternator-write-isolation` | Alternator write isolation | `only_rmw_uses_lwt` |
| `--free-tier` | Enable free tier | `False` |
| `--prometheus-proxy` | Enable Prometheus proxy | `False` |
| `--user-api` | User API interface | `CQL` |
| `--enable-dns` | Enable DNS | `False` |
| `--enable-vpc-peering` | Enable VPC peering | `False` |

### Destroy Command

| Option | Description |
|--------|-------------|
| `--name` | Cluster name (required) |
| `--force` | Skip confirmation prompt |

### Status Command

| Option | Description |
|--------|-------------|
| `--name` | Cluster name (required) |

### Info Command

| Option | Description |
|--------|-------------|
| `--name` | Cluster name (required) |

### List Command

No additional options required.

### List Clusters Command

List all clusters for a specific account from the ScyllaDB Cloud API:

```bash
./deploy-scylla-cloud.py list-clusters --account-id "your-account-id"
```

Or using environment variable:
```bash
export SCYLLA_CLOUD_ACCOUNT_ID="your-account-id"
./deploy-scylla-cloud.py list-clusters
```

**JSON output:**
```bash
./deploy-scylla-cloud.py list-clusters --format json
```

| Option | Description |
|--------|-------------|
| `--account-id` | ScyllaDB Cloud account ID (required, or set `SCYLLA_CLOUD_ACCOUNT_ID` env var) |

### Get Account Info Command

Retrieve ScyllaDB Cloud account information.

No additional options required. Only `--api-key` (or `SCYLLA_CLOUD_API_KEY`) is needed.

## How Cluster Creation Works

When you create a cluster, the tool performs the following steps:

1. **Validates** your configuration (account ID, node count, etc.)
2. **Looks up Cloud Provider ID** from the provider name (AWS/GCP) using `/deployment/cloud-providers` endpoint
3. **Looks up Region ID** from the region name using `/deployment/cloud-provider/{id}/regions` endpoint
4. **Looks up Instance Type IDs** for both ScyllaDB and vector search nodes using `/deployment/cloud-provider/{id}/region/{id}` endpoint
5. **Constructs API request** with proper field names and numeric IDs
6. **Creates cluster** via `/cluster/v1/clusters` endpoint
7. **Saves cluster info** to local state file (`~/.scylla-clusters.json`)

This automatic ID translation means you can use familiar names like "AWS" and "us-east-1" instead of looking up numeric IDs manually.

### API Request Body

The tool constructs a request body that matches the ScyllaDB Cloud API specification:

```json
{
  "name": "mycluster",
  "accountId": "12345",
  "cloudProviderId": 1,
  "regionId": 10,
  "instanceId": 42,
  "numberOfNodes": 3,
  "cidrBlock": "192.168.1.0/24",
  "broadcastType": "PUBLIC",
  "replicationFactor": 3,
  "tablets": "enforced",
  "vectorSearch": {
    "defaultNodes": 3,
    "defaultInstanceTypeId": 42
  }
}
```

**Important**: The tool uses the correct API field names (`cloudProviderId`, `regionId`, `instanceId`, `numberOfNodes`, etc.) which differ from the CLI option names for better user experience.

## Output Formats

### Text Format (Default)

Human-readable output suitable for terminal use:

```
Cluster: mycluster
ID: 12345678-1234-1234-1234-123456789abc
Status: ACTIVE
Cloud Provider: AWS
Region: us-east-1
Node Count: 3
Vector Search: Enabled
  Vector Node Count: 3
```

### JSON Format

Structured output for scripting and automation:

```bash
./deploy-scylla-cloud.py status --name mycluster --format json
```

```json
{
  "id": "12345678-1234-1234-1234-123456789abc",
  "name": "mycluster",
  "status": "ACTIVE",
  "cloudProvider": "AWS",
  "region": "us-east-1",
  "nodeCount": 3,
  "vectorSearch": {
    "enabled": true,
    "nodeCount": 3
  }
}
```

## State Management

The tool maintains a local state file at `~/.scylla-clusters.json`:

```json
{
  "clusters": {
    "mycluster": {
      "cluster_id": "12345678-1234-1234-1234-123456789abc",
      "name": "mycluster",
      "cloud_provider": "AWS",
      "region": "us-east-1",
      "created_at": "2026-01-18 10:30:00",
      "config": {
        "name": "mycluster",
        "cloudProvider": "AWS",
        "region": "us-east-1",
        "nodeCount": 3,
        "nodeType": "i4i.large",
        "vectorSearch": {
          "enabled": true,
          "nodeCount": 3,
          "nodeType": "i4i.large"
        }
      }
    }
  }
}
```

## Common Workflows

### Quick Development Cluster

```bash
# Get your account ID first (if you don't know it)
./deploy-scylla-cloud.py get-account-info

# Create minimal cluster (vector search enabled by default)
./deploy-scylla-cloud.py create --name dev-cluster --account-id "your-account-id"

# Check when ready
./deploy-scylla-cloud.py status --name dev-cluster

# Get connection info
./deploy-scylla-cloud.py info --name dev-cluster

# Clean up when done
./deploy-scylla-cloud.py destroy --name dev-cluster --force
```

### Production Cluster with Vector Search

```bash
# Create production-ready cluster
./deploy-scylla-cloud.py create \
  --name prod-vectordb \
  --account-id "your-account-id" \
  --cloud-provider AWS \
  --region us-east-1 \
  --node-count 5 \
  --node-type i4i.2xlarge \
  --vector-node-count 3 \
  --vector-node-type i4i.xlarge \
  --enable-dns \
  --enable-vpc-peering

# Monitor status until active
./deploy-scylla-cloud.py status --name prod-vectordb

# Export connection info for application
./deploy-scylla-cloud.py info --name prod-vectordb --format json > connection.json
```

### Scripting Example

```bash
#!/bin/bash

# Create cluster
./deploy-scylla-cloud.py create \
  --name test-cluster \
  --enable-vector-search \
  --format json > cluster.json

# Extract cluster ID
CLUSTER_ID=$(jq -r '.id' cluster.json)
echo "Cluster ID: $CLUSTER_ID"

# Wait for cluster to be ready
while true; do
  STATUS=$(./deploy-scylla-cloud.py status --name test-cluster --format json | jq -r '.status')
  echo "Status: $STATUS"
  
  if [ "$STATUS" = "ACTIVE" ]; then
    echo "Cluster is ready!"
    break
  fi
  
  sleep 30
done

# Get connection info
./deploy-scylla-cloud.py info --name test-cluster --format json > connection.json
```

## Error Handling

The tool follows these error handling principles:

1. **Validation Errors**: Caught early with clear messages (e.g., invalid cloud provider, region, or instance type names)
2. **API Lookup Errors**: Clear error messages when cloud provider, region, or instance type cannot be found
3. **API Errors**: HTTP errors from ScyllaDB Cloud API are displayed with response details
4. **Failed Resources**: Left in place for manual inspection (no automatic cleanup)
5. **State Consistency**: Local state always reflects attempted operations

## Instance Types

### AWS Instance Types
- `i4i.large` - 2 vCPUs, 16 GB RAM (default)
- `i4i.xlarge` - 4 vCPUs, 32 GB RAM
- `i4i.2xlarge` - 8 vCPUs, 64 GB RAM
- `i4i.4xlarge` - 16 vCPUs, 128 GB RAM

### GCP Instance Types
- `n2-highmem-2` - 2 vCPUs, 16 GB RAM
- `n2-highmem-4` - 4 vCPUs, 32 GB RAM
- `n2-highmem-8` - 8 vCPUs, 64 GB RAM
- `n2-highmem-16` - 16 vCPUs, 128 GB RAM

## Regions

### AWS Regions
- `us-east-1` (default)
- `us-west-2`
- `eu-west-1`
- `ap-southeast-1`

### GCP Regions
- `us-central1`
- `us-east1`
- `europe-west1`
- `asia-southeast1`

Check the [ScyllaDB Cloud documentation](https://cloud.docs.scylladb.com) for the latest supported regions and instance types.

## Debugging

The tool includes comprehensive debug output to help diagnose API issues:

```bash
./deploy-scylla-cloud.py create --name mycluster --account-id "your-account-id" --debug
```

With `--debug` enabled, the tool displays:
- Full request URL and headers
- Complete request body (JSON)
- Response status code
- Response headers
- Detailed response body

This is especially helpful when:
- Troubleshooting 4xx/5xx HTTP errors
- Verifying API request format
- Understanding API error messages
- Debugging configuration issues

**Example debug output:**
```
=== DEBUG: POST https://api.cloud.scylladb.com/cluster/v1/clusters ===
Headers: {
  "Content-Type": "application/json"
}
Request Body: {
  "name": "mycluster",
  "accountId": "12345",
  "cloudProviderId": 1,
  "regionId": 10,
  "instanceId": 42,
  "numberOfNodes": 3,
  "cidrBlock": "192.168.1.0/24",
  "broadcastType": "PUBLIC",
  "replicationFactor": 3,
  "tablets": "enforced",
  "vectorSearch": {
    "defaultNodes": 3,
    "defaultInstanceTypeId": 42
  }
}

Response Status: 200
Response Headers: {...}
Response Body: {...}
=== END DEBUG ===
```

## Troubleshooting

### ID Lookup Errors
```
✗ Error: Cloud provider 'XYZ' not found
✗ Error: Region 'invalid-region' not found for cloud provider ID 1
✗ Error: Instance type 'invalid-type' not found in region ID 10
```
**Solution**: Check that you're using valid cloud provider names (AWS or GCP), valid region names for your cloud provider, and valid instance types for your region. Use `--debug` to see the API responses during lookup.

### HTTP Errors
```
✗ API Error
HTTP Status: 500
```
**Solution**: Run with `--debug` to see full error details. Common causes:
- Invalid configuration parameters
- Insufficient API key permissions
- ScyllaDB Cloud service issues
- Invalid region or instance type combinations (though these are now caught during ID lookup)

### API Key Issues
```
✗ Error: API key required. Use --api-key or set SCYLLA_CLOUD_API_KEY
```
**Solution**: Set the `SCYLLA_CLOUD_API_KEY` environment variable or use `--api-key`.

### Missing Account ID
```
✗ Error: Account ID is required. Use --account-id or set SCYLLA_CLOUD_ACCOUNT_ID
```
**Solution**: Get your account ID using the `get-account-info` command:
```bash
./deploy-scylla-cloud.py get-account-info
```
Then provide it via `--account-id` or set the `SCYLLA_CLOUD_ACCOUNT_ID` environment variable. You can also find your account ID in the ScyllaDB Cloud console.

### Cluster Already Exists
```
✗ Error: Cluster 'mycluster' already exists in local state
```
**Solution**: Use a different name or destroy the existing cluster first.

### Minimum Node Count
```
✗ Error: Minimum node count is 3
```
**Solution**: ScyllaDB Cloud requires at least 3 nodes for high availability.

### HTTP Errors
```
✗ API Error: 401 Client Error: Unauthorized
```
**Solution**: Verify your API key is valid and has the necessary permissions.

## Integration with ai_agent_with_cache.py

After creating a cluster with vector search, use the connection information in the main AI agent:

```bash
# Create cluster and get connection info
./deploy-scylla-cloud.py create --name ai-cache --account-id "your-account-id"
./deploy-scylla-cloud.py info --name ai-cache --format json > connection.json

# Extract connection details
SCYLLA_HOST=$(jq -r '.connection.host' connection.json)
SCYLLA_USER=$(jq -r '.connection.username' connection.json)
SCYLLA_PASSWORD=$(jq -r '.connection.password' connection.json)

# Use with the AI agent
cd ..
./ai_agent_with_cache.py \
  --prompt "Your prompt here" \
  --with-cache scylla \
  --scylla-contact-points "$SCYLLA_HOST" \
  --scylla-user "$SCYLLA_USER" \
  --scylla-password "$SCYLLA_PASSWORD"
```

## API Reference

This tool uses the ScyllaDB Cloud REST API. For detailed API documentation, see:
- [ScyllaDB Cloud API Documentation](https://cloud.docs.scylladb.com/stable/api.html)
- [Cluster Management API](https://cloud.docs.scylladb.com/stable/api.html#tag/Cluster)

## License

[Specify your license here]
