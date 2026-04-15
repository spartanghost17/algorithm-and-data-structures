# DSA Interview Patterns & Intuition Guide

---

## 1. Arrays & Hashing

**Core Intuition:** Trade space for time. Use a HashMap to remember what you've seen.

**When to use:**

- Need O(1) lookup for previously seen elements
- Counting frequency of elements
- Finding duplicates or pairs
- Grouping items by some property

**Keywords / Giveaways:**

- "Find if exists" → HashMap lookup
- "Count occurrences" → HashMap frequency
- "Two sum / pair that equals X" → HashMap complement
- "Group by" / "anagrams" → HashMap with key transformation
- "First non-repeating" → HashMap + order
- "Subarray sum equals K" → Prefix sum + HashMap

**Pseudo-pattern:**

```python
map = {}
for each element:
    if complement/target in map:
        found answer
    map[element] = index/count
```

**Common Mistakes:**

- Forgetting to handle duplicates
- Off-by-one with indices
- Not considering negative numbers with prefix sums

---

## 2. Two Pointers

**Core Intuition:** Use two pointers to avoid nested loops. Usually reduces O(n²) → O(n).

**When to use:**

- Sorted array problems
- Finding pairs with a condition
- Comparing from both ends
- In-place array modifications
- Palindrome checking

**Keywords / Giveaways:**

- "Sorted array" → Two pointers from ends
- "Pair/triplet that sums to X" → Sort + two pointers
- "Remove duplicates in-place" → Slow/fast pointer
- "Palindrome" → Left/right moving inward
- "Container with most water" → Shrink from worse side
- "Merge two sorted" → Pointer per array

**Pseudo-pattern:**

```python
# Pattern 1: Opposite ends
left, right = 0, len-1
while left < right:
    if condition met: return answer
    if need bigger: left++
    else: right--

# Pattern 2: Same direction (slow/fast)
slow, fast = 0, 0
while fast < len:
    if valid: arr[slow] = arr[fast]; slow++
    fast++
```

**Common Mistakes:**

- Forgetting to sort first
- Infinite loops (not moving pointers)
- Wrong comparison (< vs <=)

---

## 3. Sliding Window

**Core Intuition:** Maintain a "window" that expands/contracts. Avoid recalculating from scratch.

**When to use:**

- Contiguous subarray/substring problems
- Finding max/min of all windows of size K
- Longest/shortest substring with condition

**Keywords / Giveaways:**

- "Subarray" / "Substring" → Likely sliding window
- "Contiguous" → Sliding window
- "Maximum/minimum of size K" → Fixed window
- "Longest with at most K distinct" → Variable window
- "Smallest subarray with sum ≥ X" → Variable window (shrink when valid)

**Pseudo-pattern:**

```python
# Fixed window
for i in range(len):
    add arr[i] to window
    if i >= k-1:
        record answer
        remove arr[i-k+1] from window

# Variable window (longest valid)
left = 0
for right in range(len):
    add arr[right] to window
    while window is invalid:
        remove arr[left]; left++
    update max_length

# Variable window (shortest valid)
left = 0
for right in range(len):
    add arr[right] to window
    while window is valid:
        update min_length
        remove arr[left]; left++
```

**Common Mistakes:**

- Wrong window size calculation (right - left + 1)
- Shrinking when should expand or vice versa
- Not handling empty window case

---

## 4. Binary Search

**Core Intuition:** Eliminate half the search space each step. Works on sorted/monotonic data.

**When to use:**

- Sorted array lookup
- Finding boundary (first/last occurrence)
- Search space is monotonic (if X works, all > X work)
- Optimization problems: "minimum that satisfies"

**Keywords / Giveaways:**

- "Sorted array" → Binary search
- "O(log n) required" → Binary search
- "Find minimum/maximum that satisfies" → Binary search on answer
- "Kth smallest/largest" → Binary search on value
- "Rotated sorted array" → Modified binary search
- "Peak element" → Binary search

**Pseudo-pattern:**

```python
# Standard
left, right = 0, len-1
while left <= right:
    mid = left + (right - left) // 2
    if arr[mid] == target: return mid
    if arr[mid] < target: left = mid + 1
    else: right = mid - 1

# Find first occurrence (left boundary)
while left < right:
    mid = left + (right - left) // 2
    if arr[mid] >= target: right = mid
    else: left = mid + 1

# Binary search on answer
left, right = min_possible, max_possible
while left < right:
    mid = (left + right) // 2
    if canAchieve(mid): right = mid  # find minimum
    else: left = mid + 1
```

**Common Mistakes:**

- Integer overflow: use `left + (right - left) // 2`
- Infinite loop: wrong boundary update
- Off-by-one: `<` vs `<=`, `mid` vs `mid+1`

---

## 5. Stacks

**Core Intuition:** LIFO. Track "pending" items. Great for matching pairs and "next greater" problems.

**When to use:**

- Matching parentheses/brackets
- Next greater/smaller element
- Evaluate expressions
- Undo operations
- Monotonic problems

**Keywords / Giveaways:**

- "Valid parentheses" / "balanced" → Stack
- "Next greater element" → Monotonic stack
- "Evaluate expression" → Stack for operands
- "Largest rectangle in histogram" → Monotonic stack
- "Daily temperatures" → Monotonic stack
- "Decode string" → Stack for nested structure

**Pseudo-pattern:**

```python
# Matching pairs
for char in string:
    if opening: stack.push(char)
    if closing:
        if stack.empty or not match: invalid
        stack.pop()
return stack.empty

# Next greater (monotonic decreasing stack)
result = [-1] * len
stack = []  # stores indices
for i in range(len):
    while stack and arr[i] > arr[stack.top]:
        result[stack.pop()] = arr[i]
    stack.push(i)
```

**Common Mistakes:**

- Empty stack check before pop/peek
- Storing values vs indices (usually need indices)
- Wrong monotonic direction

---

## 6. Queues

**Core Intuition:** FIFO. Process in order received. Level-by-level processing.

**When to use:**

- BFS traversal
- Level-order tree traversal
- Scheduling/ordering problems
- Sliding window maximum (monotonic deque)

**Keywords / Giveaways:**

- "Level order" → Queue + BFS
- "Shortest path unweighted" → BFS with queue
- "Process in order" → Queue
- "Sliding window maximum" → Monotonic deque
- "First come first serve" → Queue

**Pseudo-pattern:**

```python
# BFS with queue
queue = [start]
visited = {start}
while queue:
    current = queue.pop(0)  # or deque.popleft()
    process(current)
    for neighbor in getNeighbors(current):
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)

# Monotonic deque (sliding window max)
deque = []
for i in range(len):
    while deque and arr[deque.back] < arr[i]:
        deque.pop_back()
    deque.push_back(i)
    if deque.front <= i - k:
        deque.pop_front()
    if i >= k-1:
        result.add(arr[deque.front])
```

---

## 7. Linked Lists

**Core Intuition:** Pointer manipulation. Draw it out. Use dummy nodes.

**When to use:**

- In-place list manipulation
- Cycle detection
- Merge/split lists
- Reverse portions

**Keywords / Giveaways:**

- "Reverse linked list" → Three pointers (prev, curr, next)
- "Detect cycle" → Fast/slow pointers
- "Find middle" → Fast/slow pointers
- "Merge two sorted lists" → Dummy head + two pointers
- "Remove nth from end" → Two pointers with gap
- "Intersection point" → Length difference or two-pointer cycle

**Pseudo-pattern:**

```python
# Reverse
prev = null
while curr:
    next = curr.next
    curr.next = prev
    prev = curr
    curr = next
return prev

# Find middle (slow at middle when fast reaches end)
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
return slow

# Detect cycle
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast: return True
return False
```

**Common Mistakes:**

- Losing reference to nodes
- Not handling null/empty list
- Forgetting to update `.next` pointers

---

## 8. Trees (BFS & DFS)

**Core Intuition:**

- DFS: Go deep, use recursion or stack. Pre/in/post order.
- BFS: Go level-by-level, use queue.

**When to use:**

- DFS: Path problems, tree construction, serialize/deserialize
- BFS: Level-order, shortest depth, connect level nodes

**Keywords / Giveaways:**

- "Path from root to X" → DFS
- "Maximum depth" → DFS or BFS
- "Level order" / "zigzag" → BFS
- "Lowest common ancestor" → DFS
- "Validate BST" → DFS inorder
- "Serialize/deserialize" → DFS preorder or BFS

**Pseudo-pattern:**

```python
# DFS Recursive
def dfs(node):
    if not node: return base_case
    # preorder: process here
    left = dfs(node.left)
    # inorder: process here
    right = dfs(node.right)
    # postorder: process here
    return combine(left, right)

# BFS Level Order
queue = [root]
while queue:
    level_size = len(queue)
    for i in range(level_size):
        node = queue.pop(0)
        process(node)
        if node.left: queue.append(node.left)
        if node.right: queue.append(node.right)
```

**Common Mistakes:**

- Null checks
- Confusing pre/in/post order
- Not tracking level boundaries in BFS

---

## 9. Graphs (BFS & DFS)

**Core Intuition:**

- DFS: Explore as far as possible, backtrack. Use for connectivity, cycles, paths.
- BFS: Explore level by level. Use for shortest path (unweighted).

**When to use:**

- DFS: Connected components, cycle detection, topological sort, all paths
- BFS: Shortest path, minimum steps, level-by-level spread

**Keywords / Giveaways:**

- "Number of islands" / "connected components" → DFS/BFS flood fill
- "Shortest path" / "minimum steps" → BFS
- "Course schedule" / "dependencies" → Topological sort (DFS or Kahn's BFS)
- "Detect cycle" → DFS with coloring or BFS with indegree
- "All paths from A to B" → DFS backtracking
- "Clone graph" → DFS/BFS with HashMap

**Pseudo-pattern:**
2D -> 1D = (index = r \* n + c)
1D -> 2D :
r = index / n
c = index % n

n = number of numCols

```python
# DFS on graph
visited = set()
def dfs(node):
    if node in visited: return
    visited.add(node)
    for neighbor in graph[node]:
        dfs(neighbor)

# BFS shortest path
queue = [(start, 0)]  # (node, distance)
visited = {start}
while queue:
    node, dist = queue.pop(0)
    if node == target: return dist
    for neighbor in graph[node]:
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append((neighbor, dist + 1))
```

```python
# dijkstra
def dijkstra(graph, start):
    # graph = {node: [(neighbor, weight), ...]}

    # Step 1: initialize distances
    dist = {node: float('inf') for node in graph}
    dist[start] = 0

    # Step 2: min-heap (distance, node)
    pq = [(0, start)]

    while pq:
        curr_dist, node = heapq.heappop(pq)

        # Skip outdated entries
        if curr_dist > dist[node]:
            continue

        # Step 3: relax edges
        for neighbor, weight in graph[node]:
            new_dist = curr_dist + weight

            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return dist
```

```python
# Connected components
# time = O(V + E); space = O(V)
n = number of nodes in graph
g = adjacency list representing graph
count = 0
components: list[int] = [] # size n
visited = [False,...,False] # size n
findComponents():
    for i in range(n: number of nodes):
        if not visited[i]:
            count++
            dfs(i)
    return (count, components)
dfs(at):
    visited[at] = true
    components[at] = count
    for (next: g[at]):
        if not visited[next]:
            dfs(at)
```

```python
Shortest path BFS
R, C = # R = number of rows, C = numbers of columns
m = # Input character matrix of size R xC
sr, sc = # 'S' symbol row and column values
rq, cq = # Empty Row Queue (RQ) and Column Queue (CQ)

# Variables used to track the number of steps taken
move_count = 0
nodes_left_in_layer = 1
nodes_in_next_layer = 0

# Variable used to track whether the 'E' character ever gets reached during BFS
reached_end = False

# R x C matrix of false values used to track whether the node at position (i, j) has been visited

visited = ...

# North, south, east, west direction vectors
dr = [-1, 1, 0, 0]
dc = [0, 0, 1, -1]


solve():
    rq.enqueue(sr)
    cq.enqueue(sc)
    visited[sr][sc] = true
    while rq.size() > 0:
        r = rq.dequeue()
        c = cq.dequeue()
        if m[r][c] == 'E':
            reached_end = true
            break
        explore_neighbours(r, c) # just adds valid neighbors to queue
        nodes_left_in_layer--
        if nodes_left_in_layer == 0:
            nodes_in_next_layer = nodes_in_next_layer
            nodes_in_next_layer = 0
            move_count++
    if reached_end:
        return move_count
    return -1

explore_neighbours(r, c):
    for(i = 0; i < 4; i++):
        rr = r + dr[i]
        cc = c + dc[i]

        # Skip out of bounds locations
        if rr < 0 or cc < 0: continue
        if rr >= R or cc >= C: continue

        # Skip visited locations or blocked cells
        if visited[rr][cc]: continue
        if m[rr][cc] == '#': continue

        rq.enqueue(rr)
        cq.enqueue(cc)
        visited[rr][cc] = True
        nodes_in_next_layer++
```

```python
# Topological Sort (Kahn's BFS)
indegree = count incoming edges for each node
queue = [nodes with indegree 0]
order = []
while queue:
    node = queue.pop(0)
    order.append(node)
    for neighbor in graph[node]:
        indegree[neighbor]--
        if indegree[neighbor] == 0:
            queue.append(neighbor)
if len(order) != num_nodes: cycle exists
```

**Common Mistakes:**

- Forgetting visited set (infinite loops)
- Wrong adjacency list building
- Not handling disconnected components

**Graph common patterns**

# Graph Algorithm Patterns — Complete Reference

---

## 1. BFS (Breadth-First Search)

**When to use:** Finding the shortest path in an **unweighted** graph, level-order traversal, finding all nodes within a certain distance, or any problem where you need to explore neighbours layer by layer. If a problem says "minimum number of steps/moves/transformations", BFS is almost always the answer.

**Complexity:** O(V + E) time, O(V) space

```python
BFS(graph, start):
    queue = [start]
    visited = {start}
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

**Common variations:**

- **Multi-source BFS:** Push all sources into the queue at the start (e.g. "distance from any gate" problems).
- **0-1 BFS:** Use a deque; push weight-0 edges to front, weight-1 edges to back. Replaces Dijkstra when weights are only 0 or 1.

---

## 2. DFS (Depth-First Search)

**When to use:** Exploring all paths, detecting cycles, topological sorting, finding connected components, solving maze/backtracking problems, or any scenario where you need to go as deep as possible before backtracking. Preferred over BFS when you need to explore entire branches (e.g. "count all paths", "does a path exist").

**Complexity:** O(V + E) time, O(V) space (recursion stack)

```python
DFS(graph, node, visited):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            DFS(graph, neighbor, visited)
```

**Iterative version (avoids stack overflow):**

```python
DFS_Iterative(graph, start):
    stack = [start]
    visited = set()
    while stack:
        node = stack.pop()
        if node in visited: continue
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append(neighbor)
```

---

## 3. Dijkstra's Algorithm

**When to use:** Shortest path from a single source in a graph with **non-negative** weights. The go-to algorithm whenever edges have varying positive costs (e.g. road networks, weighted grids, minimum cost problems). Will not work correctly if any edge weight is negative.

**Complexity:** O((V + E) log V) with a binary heap

```python
Dijkstra(graph, start):
    dist = {node: INF for all nodes}
    dist[start] = 0
    pq = [(0, start)]                  // min-heap: (distance, node)
    while pq:
        d, u = heappop(pq)
        if d > dist[u]: continue        // stale entry, skip
        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heappush(pq, (dist[v], v))
    return dist
```

**Key insight:** The `if d > dist[u]: continue` line is the lazy deletion optimisation — it avoids needing a decrease-key operation.

---

## 4. Bellman-Ford Algorithm

**When to use:** Shortest path from a single source when **negative edge weights** exist. Also the standard way to detect **negative weight cycles**. Slower than Dijkstra, so only use it when negatives are possible. Common in problems involving currency exchange, arbitrage detection, or constraint-based shortest paths.

**Complexity:** O(V × E)

```python
BellmanFord(graph, start, V):
    dist = {node: INF for all nodes}
    dist[start] = 0
    repeat V - 1 times:
        for each edge (u, v, weight):
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
    // Negative cycle detection (optional extra pass)
    for each edge (u, v, weight):
        if dist[u] + weight < dist[v]:
            return "Negative cycle exists"
    return dist
```

---

## 5. Floyd-Warshall Algorithm

**When to use:** Finding shortest paths between **all pairs** of vertices. Best when V is small (≤ 400–500) because of O(V³) complexity. Use it when you need to answer many "what's the distance from A to B?" queries, or for transitive closure problems. Also handles negative weights (but not negative cycles).

**Complexity:** O(V³) time, O(V²) space

```python
FloydWarshall(V, edges):
    dist[i][j] = INF for all i, j
    dist[i][i] = 0 for all i
    for each edge (u, v, w):
        dist[u][v] = w
    for k in 0..V-1:           // k = intermediate node
        for i in 0..V-1:
            for j in 0..V-1:
                dist[i][j] = min(dist[i][j],
                                 dist[i][k] + dist[k][j])
```

**Key insight:** The outermost loop must be `k` (the intermediate node). Getting the loop order wrong is the most common mistake.

---

## 6. Topological Sort

**When to use:** Ordering tasks/nodes in a **Directed Acyclic Graph (DAG)** such that for every edge u → v, u comes before v. Classic for dependency resolution (build systems, course prerequisites, task scheduling). Also used to detect cycles in directed graphs — if the topological order doesn't include all nodes, a cycle exists.

**Complexity:** O(V + E)

```python
// Kahn's Algorithm (BFS-based, preferred for interview clarity)
TopSort(graph, V):
    indegree = count incoming edges per node
    queue = [nodes where indegree == 0]
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    if len(order) != V: "Cycle exists — not a DAG"
    return order
```

**DFS-based alternative:** Run DFS and append to result in post-order, then reverse. Useful when you're already doing DFS for something else.

---

## 7. Cycle Detection

**When to use:** Determining whether a graph contains a cycle. The approach differs for directed vs undirected graphs. In directed graphs, use DFS with 3-colour marking (WHITE/GRAY/BLACK). In undirected graphs, a simple DFS with parent tracking or Union-Find works.

### Directed Graph (DFS 3-Colouring)

```python
HasCycle(graph):
    color = {node: WHITE for all nodes}
    for node in graph:
        if color[node] == WHITE:
            if DFS(node): return True
    return False

DFS(u):
    color[u] = GRAY                    // currently in recursion stack
    for v in graph[u]:
        if color[v] == GRAY: return True    // back edge = cycle
        if color[v] == WHITE and DFS(v):
            return True
    color[u] = BLACK                   // fully processed
    return False
```

### Undirected Graph (Union-Find)

```python
HasCycleUndirected(edges, V):
    initialise Union-Find for V nodes
    for u, v in edges:
        if Find(u) == Find(v): return True   // same component = cycle
        Union(u, v)
    return False
```

---

## 8. Union-Find (Disjoint Set Union)

**When to use:** Dynamically tracking connected components, detecting cycles in undirected graphs, Kruskal's MST, or any problem that asks "are X and Y connected?" with incremental edge additions. Extremely efficient with path compression + union by rank — nearly O(1) amortised per operation.

**Complexity:** O(α(n)) ≈ O(1) amortised per operation

```python
parent = [i for i in range(n)]
rank = [0] * n

Find(x):
    if parent[x] != x:
        parent[x] = Find(parent[x])       // path compression
    return parent[x]

Union(x, y):
    px, py = Find(x), Find(y)
    if px == py: return False              // already connected
    if rank[px] < rank[py]: swap(px, py)   // union by rank
    parent[py] = px
    if rank[px] == rank[py]: rank[px] += 1
    return True
```

---

## 9. Kruskal's Algorithm (Minimum Spanning Tree)

**When to use:** Finding the MST when you have an **edge list**. Particularly good when the graph is sparse or when edges are naturally given as a list (e.g. "connect N cities with minimum total cable"). Relies on Union-Find internally.

**Complexity:** O(E log E) — dominated by the sort

```python
Kruskal(edges, V):
    sort edges by weight ascending
    initialise Union-Find for V nodes
    mst_cost = 0
    mst_edges = 0
    for u, v, w in edges:
        if Union(u, v):                // different components
            mst_cost += w
            mst_edges += 1
            if mst_edges == V - 1: break
    return mst_cost
```

---

## 10. Prim's Algorithm (Minimum Spanning Tree)

**When to use:** Finding the MST when you have an **adjacency list**, especially for dense graphs. Grows the MST from a starting node by always picking the cheapest edge crossing the cut. Preferred over Kruskal when the graph is dense (E ≈ V²) or when you already have adjacency list representation.

**Complexity:** O((V + E) log V) with a binary heap

```python
Prim(graph, start):
    visited = {start}
    pq = [(w, start, v) for v, w in graph[start]]
    heapify(pq)
    mst_cost = 0
    while pq and len(visited) < V:
        w, u, v = heappop(pq)
        if v in visited: continue
        visited.add(v)
        mst_cost += w
        for next_v, next_w in graph[v]:
            if next_v not in visited:
                heappush(pq, (next_w, v, next_v))
    return mst_cost
```

---

## 11. Tarjan's Algorithm (Strongly Connected Components)

**When to use:** Finding all **Strongly Connected Components** in a directed graph — maximal subsets of vertices where every vertex is reachable from every other. Used in compiler optimisation (dependency analysis), 2-SAT problems, and simplifying directed graphs into DAGs of SCCs.

**Complexity:** O(V + E)

```python
Tarjan(graph):
    idx = 0, stack = [], on_stack = set()
    disc = {}, low = {}, sccs = []

    DFS(u):
        disc[u] = low[u] = idx++
        stack.push(u)
        on_stack.add(u)
        for v in graph[u]:
            if v not in disc:
                DFS(v)
                low[u] = min(low[u], low[v])
            elif v in on_stack:
                low[u] = min(low[u], disc[v])
        // If u is root of an SCC
        if low[u] == disc[u]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == u: break
            sccs.append(scc)

    for node in graph:
        if node not in disc:
            DFS(node)
    return sccs
```

---

## 12. Articulation Points & Bridges

**When to use:** Finding **critical nodes** (articulation points) or **critical edges** (bridges) whose removal disconnects the graph. Used in network reliability analysis, identifying single points of failure, and connectivity problems. Only applicable to **undirected** graphs.

**Complexity:** O(V + E)

```python
FindBridgesAndArticulationPoints(graph):
    timer = 0
    disc = {}, low = {}
    bridges = [], points = set()

    DFS(u, parent):
        disc[u] = low[u] = timer++
        children = 0
        for v in graph[u]:
            if v not in disc:
                children += 1
                DFS(v, u)
                low[u] = min(low[u], low[v])
                // Bridge: no back edge from v's subtree to u or above
                if low[v] > disc[u]:
                    bridges.append((u, v))
                // Articulation point (non-root)
                if parent != -1 and low[v] >= disc[u]:
                    points.add(u)
            elif v != parent:
                low[u] = min(low[u], disc[v])
        // Articulation point (root with 2+ children)
        if parent == -1 and children > 1:
            points.add(u)

    for node in graph:
        if node not in disc:
            DFS(node, -1)
    return bridges, points
```

**Key distinction:**

- **Bridge:** `low[v] > disc[u]` (strictly greater — no way back to u at all)
- **Articulation point:** `low[v] >= disc[u]` (no way back above u)

---

## 13. Bipartite Check (2-Colouring / Graph Colouring)

**When to use:** Checking if a graph can be split into two groups where no two nodes in the same group share an edge. Equivalent to checking if the graph has no odd-length cycles. Used in matching problems, scheduling (two shifts), and determining if a graph is 2-colourable.

**Complexity:** O(V + E)

```python
IsBipartite(graph):
    color = {}
    for node in graph:
        if node in color: continue
        queue = [node]
        color[node] = 0
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if v not in color:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False       // odd cycle found
    return True
```

---

## 14. Max Flow (Edmonds-Karp / BFS-based Ford-Fulkerson)

**When to use:** Finding the maximum flow from a source to a sink in a flow network. Applies to problems involving: maximum matching in bipartite graphs, minimum cut (by Max-Flow Min-Cut theorem), edge-disjoint paths, resource allocation, and network capacity. If a problem involves "maximum number of X that can flow/be assigned", think max flow.

**Complexity:** O(V × E²) for Edmonds-Karp

```python
MaxFlow(graph, source, sink):
    build residual graph (copy of capacities)
    max_flow = 0
    while True:
        // BFS to find augmenting path in residual graph
        parent = BFS(residual, source, sink)
        if no path found: break

        // Find bottleneck along the path
        path_flow = INF
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual[u][v])
            v = u

        // Update residual capacities (forward & backward)
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= path_flow
            residual[v][u] += path_flow
            v = u

        max_flow += path_flow
    return max_flow
```

**Related:** Min-Cut = Max-Flow. After running max flow, nodes reachable from source in the residual graph form one side of the minimum cut.

---

## 15. A\* Search

**When to use:** Shortest path when you have a **heuristic** function that estimates the remaining distance to the goal. Faster than Dijkstra because it prioritises nodes that look closer to the target. Common in grid pathfinding, game AI, and navigation. Requires an **admissible** heuristic (never overestimates) for optimality.

**Complexity:** Depends on heuristic quality; worst case O((V + E) log V), best case much better than Dijkstra

```python
AStar(graph, start, goal, h):
    g = {start: 0}                     // actual cost from start
    pq = [(h(start), start)]           // priority = g + h (f-score)
    parent = {}
    while pq:
        f, u = heappop(pq)
        if u == goal:
            return reconstruct_path(parent, goal)
        for v, w in graph[u]:
            tentative = g[u] + w
            if tentative < g.get(v, INF):
                g[v] = tentative
                parent[v] = u
                heappush(pq, (tentative + h(v), v))
    return "No path"
```

**Common heuristics:**

- **Grid with 4-directional movement:** Manhattan distance `|x1-x2| + |y1-y2|`
- **Grid with 8-directional movement:** Chebyshev distance `max(|x1-x2|, |y1-y2|)`
- **Euclidean space:** Straight-line distance

---

## Quick Reference Table

| Problem                               | Algorithm                     | Key Condition                  |
| ------------------------------------- | ----------------------------- | ------------------------------ |
| Unweighted shortest path              | BFS                           | No weights or all weights = 1  |
| Weighted shortest path (no negatives) | Dijkstra                      | Non-negative weights           |
| Weighted shortest path (negatives OK) | Bellman-Ford                  | Negative weights possible      |
| All-pairs shortest path               | Floyd-Warshall                | Small V (≤ ~500)               |
| Minimum spanning tree (edge list)     | Kruskal                       | Sparse graph / edge list input |
| Minimum spanning tree (adj list)      | Prim                          | Dense graph / adj list input   |
| Task ordering / dependencies          | Topological Sort              | DAG only                       |
| Cycle detection (directed)            | DFS 3-colouring               | Directed graph                 |
| Cycle detection (undirected)          | Union-Find or DFS             | Undirected graph               |
| Dynamic connectivity                  | Union-Find                    | Incremental edge additions     |
| Strongly connected components         | Tarjan / Kosaraju             | Directed graph                 |
| Critical nodes / edges                | Articulation Points / Bridges | Undirected graph               |
| 2-colourability / odd cycle check     | Bipartite BFS                 | Undirected graph               |
| Maximum flow / minimum cut            | Edmonds-Karp                  | Flow network with capacities   |
| Heuristic-guided pathfinding          | A\*                           | Admissible heuristic available |

---

## Decision Flowchart

```
Is it a shortest path problem?
├── Yes
│   ├── Unweighted? → BFS
│   ├── Weighted, no negatives? → Dijkstra
│   ├── Negative weights? → Bellman-Ford
│   ├── All pairs needed? → Floyd-Warshall
│   └── Have a heuristic? → A*
├── Minimum spanning tree?
│   ├── Edge list? → Kruskal
│   └── Adjacency list / dense? → Prim
├── Ordering / scheduling?
│   └── Topological Sort (check it's a DAG)
├── Connectivity?
│   ├── Static components → DFS / BFS
│   ├── Dynamic (adding edges) → Union-Find
│   └── Directed → Tarjan (SCCs)
├── Cycle detection?
│   ├── Directed → DFS 3-colouring or TopSort
│   └── Undirected → Union-Find or DFS + parent
├── Network flow / matching?
│   └── Max Flow (Edmonds-Karp)
└── Critical infrastructure?
    └── Bridges / Articulation Points
```

---

## 10. Heaps & Priority Queues

**Core Intuition:** Efficiently get min/max. O(log n) insert/remove, O(1) peek.

**When to use:**

- Need repeated access to min/max
- Top K problems
- Merge K sorted lists/arrays
- Scheduling by priority

**Keywords / Giveaways:**

- "Kth largest/smallest" → Min/max heap of size K
- "Top K frequent" → Heap
- "Merge K sorted" → Min heap
- "Median of stream" → Two heaps
- "Meeting rooms" / "intervals" → Min heap by end time
- "Dijkstra's shortest path" → Min heap

**Pseudo-pattern:**

```python
# Kth largest (use MIN heap of size k)
heap = []
for num in nums:
    heappush(heap, num)
    if len(heap) > k:
        heappop(heap)
return heap[0]  # kth largest

# Top K frequent
count = Counter(nums)
heap = []
for num, freq in count.items():
    heappush(heap, (freq, num))
    if len(heap) > k:
        heappop(heap)
return [x[1] for x in heap]

# Merge K sorted lists
heap = [(list.val, i, list) for i, list in enumerate(lists) if list]
heapify(heap)
dummy = curr = ListNode(0)
while heap:
    val, i, node = heappop(heap)
    curr.next = node
    curr = curr.next
    if node.next:
        heappush(heap, (node.next.val, i, node.next))
return dummy.next
```

**Common Mistakes:**

- Min vs max heap confusion (negate values for max heap in Python)
- Heap of size K vs heap of all elements
- Tuple comparison issues

---

## 11. Backtracking

**Core Intuition:** Try all possibilities, undo and try next. Systematic brute force with pruning.

**When to use:**

- Generate all combinations/permutations/subsets
- Constraint satisfaction (Sudoku, N-Queens)
- Path finding with constraints
- Word search

**Keywords / Giveaways:**

- "All possible" / "generate all" → Backtracking
- "Permutations" / "combinations" / "subsets" → Backtracking
- "Sudoku" / "N-Queens" → Backtracking with constraints
- "Word search in grid" → Backtracking DFS
- "Palindrome partitioning" → Backtracking

**Pseudo-pattern:**

```python
def backtrack(current_state, choices):
    if goal_reached:
        result.append(copy of current_state)
        return

    for choice in choices:
        if is_valid(choice):
            make_choice(current_state, choice)      # DO
            backtrack(current_state, remaining)     # RECURSE
            undo_choice(current_state, choice)      # UNDO

# Subsets
def backtrack(start, current):
    result.append(current.copy())
    for i in range(start, len(nums)):
        current.append(nums[i])
        backtrack(i + 1, current)
        current.pop()

# Permutations
def backtrack(current):
    if len(current) == len(nums):
        result.append(current.copy())
        return
    for num in nums:
        if num not in current:
            current.append(num)
            backtrack(current)
            current.pop()
```

**Common Mistakes:**

- Forgetting to undo the choice
- Copying reference instead of value
- Wrong pruning conditions

---

## Quick Reference Cheat Sheet

| Problem Pattern          | Data Structure | Time Complexity |
| ------------------------ | -------------- | --------------- |
| Lookup/count seen before | HashMap        | O(1)            |
| Sorted + find pair       | Two Pointers   | O(n)            |
| Contiguous subarray      | Sliding Window | O(n)            |
| Sorted + find element    | Binary Search  | O(log n)        |
| Matching/nesting         | Stack          | O(n)            |
| Level-order/shortest     | Queue + BFS    | O(V+E)          |
| All paths/connectivity   | DFS            | O(V+E)          |
| Repeated min/max         | Heap           | O(log n) per op |
| Generate all X           | Backtracking   | O(2^n) or O(n!) |

---

# Interview Prep Plan

## 1-2 Week Intensive Plan

### Week 1: Foundations (Core Patterns)

**Day 1-2: Arrays, Hashing, Two Pointers**

- [ ] Two Sum (Easy) - HashMap basics
- [ ] Valid Anagram (Easy) - Frequency counting
- [ ] Group Anagrams (Medium) - HashMap with key transformation
- [ ] Two Sum II Sorted (Medium) - Two pointers
- [ ] 3Sum (Medium) - Sort + two pointers
- [ ] Container With Most Water (Medium) - Two pointers

**Day 3-4: Sliding Window & Binary Search**

- [ ] Best Time to Buy and Sell Stock (Easy) - Track min
- [ ] Longest Substring Without Repeating (Medium) - Variable window
- [ ] Minimum Window Substring (Hard) - Variable window
- [ ] Binary Search (Easy) - Standard template
- [ ] Search in Rotated Sorted Array (Medium)
- [ ] Find Minimum in Rotated Array (Medium)

**Day 5-6: Stacks & Linked Lists**

- [ ] Valid Parentheses (Easy)
- [ ] Min Stack (Medium)
- [ ] Daily Temperatures (Medium) - Monotonic stack
- [ ] Reverse Linked List (Easy)
- [ ] Merge Two Sorted Lists (Easy)
- [ ] Linked List Cycle (Easy)
- [ ] Remove Nth Node From End (Medium)

**Day 7: Review + Practice**

- Review all patterns from week 1
- Redo any problems you struggled with
- Do 2-3 random problems mixing patterns

### Week 2: Trees, Graphs, Advanced

**Day 8-9: Trees**

- [ ] Maximum Depth of Binary Tree (Easy)
- [ ] Same Tree (Easy)
- [ ] Invert Binary Tree (Easy)
- [ ] Binary Tree Level Order Traversal (Medium) - BFS
- [ ] Validate BST (Medium) - DFS inorder
- [ ] Lowest Common Ancestor (Medium)
- [ ] Binary Tree Right Side View (Medium)

**Day 10-11: Graphs**

- [ ] Number of Islands (Medium) - Flood fill
- [ ] Clone Graph (Medium)
- [ ] Course Schedule (Medium) - Topological sort
- [ ] Pacific Atlantic Water Flow (Medium)
- [ ] Graph Valid Tree (Medium)
- [ ] Number of Connected Components (Medium)

**Day 12-13: Heaps & Backtracking**

- [ ] Kth Largest Element (Medium)
- [ ] Top K Frequent Elements (Medium)
- [ ] Merge K Sorted Lists (Hard)
- [ ] Subsets (Medium)
- [ ] Permutations (Medium)
- [ ] Combination Sum (Medium)
- [ ] Word Search (Medium)

**Day 14: Mock Interviews + Review**

- Do 2-3 timed mock interviews (45 min each)
- Review weak areas
- Practice explaining your thought process out loud

---

## Daily Study Structure (3-4 hours)

```
30 min - Review pattern theory
90 min - Solve 3 problems (Easy/Medium/Medium)
30 min - Review solutions, note patterns
30 min - Spaced repetition of old problems
```

**Problem Solving Approach:**

1. Read problem, identify pattern (2 min)
2. Think through approach, write pseudocode (5-8 min)
3. Code solution (15-20 min)
4. Test with examples (5 min)
5. If stuck > 20 min, look at hints/solution
6. Always understand WHY the solution works

---

## Long-Term Goals (1-3 months)

### Month 1: Build Foundation

- Complete NeetCode 150 or Blind 75
- Master all 11 patterns above
- Solve 80-100 problems total
- Focus on Medium difficulty

### Month 2: Deepen & Speed Up

- Tackle Hard problems in weak areas
- Reduce average solve time (target: 20-25 min for medium)
- Learn dynamic programming basics
- Practice mock interviews weekly
- Solve 50-70 more problems

### Month 3: Polish & Interview Ready

- Weekly mock interviews
- System design basics (for senior roles)
- Behavioral question prep (STAR method)
- Company-specific practice (LeetCode company tags)
- Review and maintain skills

---

## Difficulty Progression

```
Week 1-2:  70% Easy, 30% Medium
Week 3-4:  30% Easy, 60% Medium, 10% Hard
Month 2+:  10% Easy, 60% Medium, 30% Hard
```

---

# Who Asks These Questions?

## Companies by DSA Difficulty

### Heavy DSA (LeetCode Medium-Hard)

- **FAANG**: Google, Meta, Amazon, Apple, Netflix
- **Big Tech**: Microsoft, Uber, Lyft, Airbnb, LinkedIn
- **Trading/Finance**: Jane Street, Two Sigma, Citadel, HRT
- **Unicorns**: Stripe, Databricks, Snowflake, OpenAI

**Expect:** 2-4 coding rounds, 45-60 min each, LeetCode Medium-Hard

### Moderate DSA (LeetCode Easy-Medium)

- **Mid-size Tech**: Salesforce, Adobe, Oracle, VMware
- **Enterprise**: IBM, Cisco, Intel, PayPal
- **Fast-growing**: Coinbase, Shopify, DoorDash, Instacart
- **Banks**: Goldman Sachs, JPMorgan, Capital One

**Expect:** 1-3 coding rounds, mix of Easy-Medium, some system design

### Light DSA (Practical Focus)

- **Startups**: Most early-stage companies
- **Agencies**: Consulting firms, dev agencies
- **Non-tech Companies**: Retail, healthcare, media tech teams
- **Government/Defense**: Contractors, public sector

**Expect:** Take-home projects, practical coding, debugging exercises

---

## Reality Check

| Company Type | DSA Weight | What Else Matters           |
| ------------ | ---------- | --------------------------- |
| FAANG        | 70%        | System design, behavioral   |
| Unicorns     | 60%        | System design, culture fit  |
| Mid-size     | 50%        | Experience, projects        |
| Startups     | 30%        | Practical skills, ship fast |
| Enterprise   | 40%        | Domain knowledge, stability |

---

## Should You Prep DSA?

**YES if:**

- Targeting top tech companies
- New grad / early career
- Switching from non-tech
- Want to maximize compensation

**LESS CRITICAL if:**

- Targeting startups
- Senior with strong portfolio
- Specialized roles (DevOps, Data Eng)
- Non-tech industry

**Bottom Line:** DSA prep is highest ROI for FAANG/Big Tech. For other companies, balance with practical projects and system design.

---

## Final Tips

1. **Quality > Quantity** - Understanding 100 problems deeply beats grinding 300 mindlessly
2. **Pattern Recognition** - Learn to identify which pattern fits, not just memorize solutions
3. **Talk Out Loud** - Practice explaining your thought process
4. **Time Yourself** - Real interviews have time pressure
5. **Review Mistakes** - Keep a log of errors and revisit
6. **Take Breaks** - Burnout hurts performance
7. **Mock Interviews** - Do at least 5-10 before real interviews

Good luck! 🚀
