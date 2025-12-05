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
