# Implement Graph Visualization API

I will create a new API endpoint to retrieve the full knowledge graph (nodes and relationships) from Neo4j, optimized for visualization in a "Memory UI".

## 1. Define Data Models

I will add new Pydantic models to `backend/src/api/v1/graph/schemas.py` to structure the graph data for the frontend.

* **GraphNode**: Represents a node with `id` (Neo4j elementId), `labels`, and `properties`.

* **GraphRelationship**: Represents a link with `id`, `source` (start node ID), `target` (end node ID), `type`, and `properties`.

* **GraphVisualizationResponse**: The main response model containing lists of `nodes` and `links`.

## 2. Create API Endpoint

I will add a new `GET` endpoint to `backend/src/api/v1/graph/routes.py`.

* **Path**: `/graph/visualization`

* **Parameters**: `limit` (int, default=1000) to control the graph size.

* **Logic**:

  1. Connect to Neo4j using the existing `Neo4jClient`.
  2. Execute a Cypher query to fetch nodes and relationships:

     ```cypher
     MATCH (n)
     OPTIONAL MATCH (n)-[r]->(m)
     RETURN n, r, m
     LIMIT $limit
     ```
  3. Parse the results to extract unique nodes and relationships, ensuring no duplicates.
  4. Map the data to the new Pydantic models, using Neo4j's internal `element_id` as the unique identifier for visualization libraries.

## 3. Verification

* I will verify the code changes by checking for syntax errors.

* You can verify the functionality by calling the endpoint and connecting it to your Memory UI.

