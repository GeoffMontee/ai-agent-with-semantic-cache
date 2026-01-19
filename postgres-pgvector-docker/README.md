# PostgreSQL with pgvector Docker Deployment

A command-line tool for managing local PostgreSQL instances with pgvector extension using Docker Compose.

## Features

- 🐘 **PostgreSQL with pgvector**: Runs official pgvector/pgvector Docker images
- 🚀 **Easy Setup**: Start a PostgreSQL instance with one command
- 🔄 **Multiple Instances**: Support for multiple named instances on different ports
- 💾 **Data Persistence**: Uses Docker named volumes to preserve data
- 🔧 **Auto-initialization**: Automatically creates pgvector extension on first start
- 📊 **Status Monitoring**: Check container status and health
- 🔌 **Connection Info**: Retrieve connection details easily
- 📝 **Log Viewing**: View container logs with follow support

## Requirements

- Docker and Docker Compose installed and running
- Python 3.8+
- `docker` Python package

## Installation

1. **Install Docker**: Make sure Docker Desktop (macOS/Windows) or Docker Engine (Linux) is installed and running

2. **Install Python dependencies**:
```bash
cd postgres-pgvector-docker
pip install -r requirements.txt
```

3. **Make script executable**:
```bash
chmod +x deploy-pgvector-docker.py
```

## Usage

### Start PostgreSQL

**Basic start (default configuration):**
```bash
./deploy-pgvector-docker.py start
```

This creates a PostgreSQL 18 instance with:
- Container name: `pgvector-local`
- Port: 5432
- User: `postgres`
- Password: `postgres`
- Database: `postgres`
- pgvector extension automatically installed

**Custom configuration:**
```bash
./deploy-pgvector-docker.py start \
  --name my-pgvector \
  --postgres-version 17 \
  --port 5433 \
  --user myuser \
  --password mypassword \
  --database mydb
```

**Multiple instances:**
```bash
# Instance 1 on port 5432
./deploy-pgvector-docker.py start --name pgvector-1 --port 5432

# Instance 2 on port 5433
./deploy-pgvector-docker.py start --name pgvector-2 --port 5433
```

### Check Status

```bash
./deploy-pgvector-docker.py status --name pgvector-local
```

Output:
```
Container 'pgvector-local':
  Status: running
  Health: healthy
  Port: 0.0.0.0:5432 -> 5432/tcp
  Volume: pgvector-local-data -> /var/lib/postgresql/data
```

### Get Connection Information

```bash
./deploy-pgvector-docker.py info --name pgvector-local
```

Output:
```
Connection Information for 'pgvector-local':
  Status: running
  Host: localhost
  Port: 5432
  User: postgres
  Database: postgres
  Password: postgres

Connection string:
  postgresql://postgres:postgres@localhost:5432/postgres
```

**JSON format:**
```bash
./deploy-pgvector-docker.py info --name pgvector-local --format json
```

### View Logs

**View last 100 lines:**
```bash
./deploy-pgvector-docker.py logs --name pgvector-local
```

**Follow logs (like `tail -f`):**
```bash
./deploy-pgvector-docker.py logs --name pgvector-local --follow
```

**View last 50 lines:**
```bash
./deploy-pgvector-docker.py logs --name pgvector-local --tail 50
```

### Stop Container

Stops the container but preserves data:
```bash
./deploy-pgvector-docker.py stop --name pgvector-local
```

### Restart Container

```bash
./deploy-pgvector-docker.py restart --name pgvector-local
```

### Destroy Container

**Destroy container but keep data:**
```bash
./deploy-pgvector-docker.py destroy --name pgvector-local
```

**Destroy container and remove data (WARNING: deletes all data):**
```bash
./deploy-pgvector-docker.py destroy --name pgvector-local --remove-volumes
```

**Skip confirmation:**
```bash
./deploy-pgvector-docker.py destroy --name pgvector-local --force
```

## Command Reference

### Global Options

| Option | Description | Default |
|--------|-------------|---------|
| `--debug` | Enable debug output with full Docker commands | `False` |
| `--format` | Output format: `text` or `json` | `text` |

### Start Command

| Option | Description | Default |
|--------|-------------|---------|
| `--name` | Container name | `pgvector-local` |
| `--postgres-version` | PostgreSQL version (12-18) | `18` |
| `--user` | PostgreSQL user | `postgres` |
| `--password` | PostgreSQL password | `postgres` |
| `--database` | PostgreSQL database name | `postgres` |
| `--port` | Host port for PostgreSQL | `5432` |

### Stop Command

| Option | Description | Default |
|--------|-------------|---------|
| `--name` | Container name | `pgvector-local` |

### Restart Command

| Option | Description | Default |
|--------|-------------|---------|
| `--name` | Container name | `pgvector-local` |

### Destroy Command

| Option | Description | Default |
|--------|-------------|---------|
| `--name` | Container name | `pgvector-local` |
| `--remove-volumes` | Also remove data volumes | `False` |
| `--force` | Skip confirmation prompt | `False` |

### Status Command

| Option | Description | Default |
|--------|-------------|---------|
| `--name` | Container name | `pgvector-local` |

### Info Command

| Option | Description | Default |
|--------|-------------|---------|
| `--name` | Container name | `pgvector-local` |

### Logs Command

| Option | Description | Default |
|--------|-------------|---------|
| `--name` | Container name | `pgvector-local` |
| `--tail` | Number of lines from end | `100` |
| `--follow` / `-f` | Follow log output | `False` |

## Connecting to PostgreSQL

### Using psql

```bash
psql postgresql://postgres:postgres@localhost:5432/postgres
```

### Using Python (psycopg2)

```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    user="postgres",
    password="postgres",
    database="postgres"
)

# Verify pgvector is installed
cur = conn.cursor()
cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
print(cur.fetchone())  # ('vector', '0.8.0')
```

### Using pgvector

```python
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    user="postgres",
    password="postgres",
    database="postgres"
)

register_vector(conn)

cur = conn.cursor()
cur.execute("CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3))")
cur.execute("INSERT INTO items (embedding) VALUES (%s)", ([1, 2, 3],))
cur.execute("SELECT * FROM items ORDER BY embedding <-> %s LIMIT 1", ([3, 1, 2],))
print(cur.fetchone())
```

## Data Persistence

Data is stored in Docker named volumes:
- Volume name format: `{container-name}-data`
- Default volume: `pgvector-local-data`
- Data persists across container restarts and removals (unless `--remove-volumes` is used)

To view volumes:
```bash
docker volume ls | grep pgvector
```

To inspect a volume:
```bash
docker volume inspect pgvector-local-data
```

## Troubleshooting

### Port Already in Use

If you see an error that the port is already in use:
1. Use a different port: `--port 5433`
2. Stop the conflicting container
3. Or use a different container name

### Container Won't Start

Check logs for errors:
```bash
./deploy-pgvector-docker.py logs --name pgvector-local
```

Enable debug mode:
```bash
./deploy-pgvector-docker.py --debug start --name pgvector-local
```

### Check Docker Status

Make sure Docker is running:
```bash
docker ps
```

### Reset Everything

To completely remove a container and its data:
```bash
./deploy-pgvector-docker.py destroy --name pgvector-local --remove-volumes --force
```

## Architecture

### Components

1. **docker-compose.yml**: Base Docker Compose configuration
   - Uses official pgvector/pgvector images
   - Configures volumes, ports, and health checks
   - Supports environment variable customization

2. **init.sql**: Initialization script
   - Automatically creates pgvector extension
   - Runs on first container start

3. **deploy-pgvector-docker.py**: Python management script
   - Manages container lifecycle
   - Queries Docker API for container state
   - No separate state file needed

4. **Docker SDK**: Python docker package
   - Used to query container status
   - Check port availability
   - Retrieve container details

### Design Decisions

- **Named Volumes**: More portable than bind mounts, managed by Docker
- **No State File**: Container state queried directly from Docker
- **Multiple Instances**: Support different ports and container names
- **Auto-initialization**: pgvector extension created automatically
- **Health Checks**: Built-in PostgreSQL health monitoring

## Integration with Main Tool

This tool can provide local PostgreSQL instances for development/testing of the main AI agent:

1. Start PostgreSQL: `./deploy-pgvector-docker.py start`
2. Get connection details: `./deploy-pgvector-docker.py info`
3. Use with AI agent for local semantic caching tests

## Version Support

Supported PostgreSQL versions:
- PostgreSQL 12 with pgvector
- PostgreSQL 13 with pgvector
- PostgreSQL 14 with pgvector
- PostgreSQL 15 with pgvector
- PostgreSQL 16 with pgvector
- PostgreSQL 17 with pgvector
- PostgreSQL 18 with pgvector (default)

The tool uses the official pgvector/pgvector Docker images with the format `pgvector/pgvector:pg{version}`.
