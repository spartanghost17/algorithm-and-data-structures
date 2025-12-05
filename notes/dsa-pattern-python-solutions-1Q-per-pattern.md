# DSA Patterns: Problem Statements, Python Solutions & Intuition

---

## 1. Arrays & Hashing

### Problem: Two Sum

**Statement:** Given an array of integers `nums` and an integer `target`, return the indices of the two numbers such that they add up to target. You may assume each input has exactly one solution.

```
Input: nums = [2, 7, 11, 15], target = 9
Output: [0, 1]
Explanation: nums[0] + nums[1] = 2 + 7 = 9
```

### Python Solution

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    seen = {}  # value -> index

    for i, num in enumerate(nums):
        complement = target - num

        if complement in seen:
            return [seen[complement], i]

        seen[num] = i

    return []
```

### Keywords That Indicated This Solution

- **"Two numbers that add up to"** → Need to find a pair → HashMap for O(1) lookup
- **"Return indices"** → Store index in dict, not just existence
- **"Exactly one solution"** → Don't need to handle duplicates/multiple answers

### Intuition

The brute force is O(n²): check every pair. But notice: if we need `a + b = target`, then when we see `a`, we're really looking for `target - a`. Instead of scanning the whole array for the complement, we store everything we've seen in a dict. When we reach element `b`, we check if `target - b` exists in our dict. This reduces O(n²) → O(n).

**Mental Model:** "I've seen these numbers before, is my partner already waiting for me?"

---

## 2. Two Pointers

### Problem: Container With Most Water

**Statement:** Given an array `height` where `height[i]` is the height of a vertical line at position `i`, find two lines that together with the x-axis forms a container that holds the most water.

```
Input: height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
Output: 49
Explanation: Lines at index 1 (height 8) and index 8 (height 7) form container with area = 7 * 7 = 49
```

### Python Solution

```python
def max_area(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    max_water = 0

    while left < right:
        # Calculate current area
        width = right - left
        h = min(height[left], height[right])
        area = width * h
        max_water = max(max_water, area)

        # Move the pointer with smaller height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_water
```

### Keywords That Indicated This Solution

- **"Two lines"** → Two pointers to select two positions
- **"Most/maximum"** → Optimization problem
- **"Container"** → Width (distance) × Height (minimum of two)
- **Array without explicit "sorted"** → Two pointers from ends, greedy shrinking

### Intuition

Start with the widest container (pointers at both ends). The only way to potentially get more water is to find a taller line. Moving the taller pointer inward can never help (width decreases, height can't increase past the shorter line). So always move the shorter pointer hoping to find something taller.

**Mental Model:** "I have maximum width. I can only improve by trading width for height. Always sacrifice the weaker side."

---

## 3. Sliding Window

### Problem: Longest Substring Without Repeating Characters

**Statement:** Given a string `s`, find the length of the longest substring without repeating characters.

```
Input: s = "abcabcbb"
Output: 3
Explanation: "abc" is the longest substring without repeating characters.
```

### Python Solution (HashSet)

```python
def length_of_longest_substring(s: str) -> int:
    window = set()
    left = 0
    max_length = 0

    for right in range(len(s)):
        # Shrink window until no duplicate
        while s[right] in window:
            window.remove(s[left])
            left += 1

        # Add current character
        window.add(s[right])

        # Update max
        max_length = max(max_length, right - left + 1)

    return max_length
```

### Python Solution (Dict - Optimized)

```python
def length_of_longest_substring(s: str) -> int:
    last_seen = {}  # char -> last index seen
    left = 0
    max_length = 0

    for right, char in enumerate(s):
        # If char seen and within current window, jump left pointer
        if char in last_seen and last_seen[char] >= left:
            left = last_seen[char] + 1

        last_seen[char] = right
        max_length = max(max_length, right - left + 1)

    return max_length
```

### Keywords That Indicated This Solution

- **"Substring"** → Contiguous → Sliding window
- **"Longest"** → Variable-size window, expand and track max
- **"Without repeating"** → Constraint that determines when to shrink

### Intuition

We want contiguous characters (substring), so sliding window fits. Expand the right pointer to include more characters. When we hit a duplicate, the window is invalid—shrink from the left until valid again. Track the maximum valid window size seen.

**Mental Model:** "Grow my window greedily. When it breaks the rule, shrink from left until it's valid again."

---

## 4. Binary Search

### Problem: Search in Rotated Sorted Array

**Statement:** Given a rotated sorted array `nums` and a `target`, return the index of target or -1 if not found. The array was originally sorted in ascending order, then rotated at some pivot.

```
Input: nums = [4, 5, 6, 7, 0, 1, 2], target = 0
Output: 4
```

### Python Solution

```python
def search(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid

        # Determine which half is sorted
        if nums[left] <= nums[mid]:
            # Left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
```

### Keywords That Indicated This Solution

- **"Sorted"** (even if rotated) → Binary search variant
- **"Search"** / **"Find"** → Looking for specific element
- **"O(log n)"** implied by interview context → Must be binary search
- **"Rotated"** → Modified binary search, find sorted half first

### Intuition

In a rotated sorted array, at least one half (left or right of mid) is always properly sorted. First determine which half is sorted by comparing `nums[left]` with `nums[mid]`. Then check if target lies within the sorted half's range. If yes, search there; otherwise, search the other half.

**Mental Model:** "One side is always 'normal'. Figure out which side is sorted, then decide if my target could be hiding there."

---

## 5. Stack

### Problem: Daily Temperatures

**Statement:** Given an array `temperatures`, return an array `answer` where `answer[i]` is the number of days you have to wait after the i-th day to get a warmer temperature. If no future day is warmer, put 0.

```
Input: temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
Output: [1, 1, 4, 2, 1, 1, 0, 0]
```

### Python Solution

```python
def daily_temperatures(temperatures: list[int]) -> list[int]:
    n = len(temperatures)
    answer = [0] * n
    stack = []  # stores indices

    for i in range(n):
        # Pop all days that found their warmer day (today)
        while stack and temperatures[i] > temperatures[stack[-1]]:
            prev_day = stack.pop()
            answer[prev_day] = i - prev_day

        # Push current day (waiting for a warmer day)
        stack.append(i)

    return answer
```

### Keywords That Indicated This Solution

- **"Next greater"** / **"next warmer"** → Classic monotonic stack pattern
- **"Wait until"** / **"how many days until"** → Looking forward, stack stores pending items
- **"For each element, find the next X"** → Stack to track unresolved elements

### Intuition

Each day is "waiting" for a warmer day. Push days onto the stack. When a warmer day comes, it resolves all the cooler days waiting on the stack. Pop them and record how long they waited. The stack maintains a decreasing order of temperatures (monotonic decreasing stack).

**Mental Model:** "Each day goes into a waiting room (stack). When a hotter day arrives, it 'answers' everyone in the waiting room who was cooler."

---

## 6. Queue / BFS

### Problem: Rotting Oranges

**Statement:** In a grid, 0 = empty, 1 = fresh orange, 2 = rotten orange. Every minute, fresh oranges adjacent to rotten ones become rotten. Return the minimum minutes until no fresh orange remains, or -1 if impossible.

```
Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4
```

### Python Solution

```python
from collections import deque

def oranges_rotting(grid: list[list[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh_count = 0

    # Initialize: find all rotten oranges and count fresh ones
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c))
            elif grid[r][c] == 1:
                fresh_count += 1

    if fresh_count == 0:
        return 0

    minutes = 0
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    while queue:
        rotten_this_minute = False

        for _ in range(len(queue)):  # Process current level
            row, col = queue.popleft()

            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc

                if (0 <= new_row < rows and
                    0 <= new_col < cols and
                    grid[new_row][new_col] == 1):

                    grid[new_row][new_col] = 2  # Mark rotten
                    queue.append((new_row, new_col))
                    fresh_count -= 1
                    rotten_this_minute = True

        if rotten_this_minute:
            minutes += 1

    return minutes if fresh_count == 0 else -1
```

### Keywords That Indicated This Solution

- **"Minimum minutes/time"** → Shortest path/time → BFS
- **"Spreads"** / **"adjacent"** → Level-by-level propagation → BFS
- **"Grid"** + **"spreading"** → Multi-source BFS
- **"Simultaneously"** → All sources start at once (multi-source BFS)

### Intuition

This is a multi-source BFS. All rotten oranges start spreading simultaneously—put them all in the queue initially. Each BFS level = 1 minute. Process level by level, counting minutes. BFS guarantees minimum time because it explores all cells at distance K before distance K+1.

**Mental Model:** "Rot spreads like a wave. All rotten oranges are the wave's starting points. Each ripple outward = 1 minute."

---

## 7. Linked List

### Problem: Reverse Linked List

**Statement:** Given the head of a singly linked list, reverse the list and return the new head.

```
Input: 1 -> 2 -> 3 -> 4 -> 5 -> null
Output: 5 -> 4 -> 3 -> 2 -> 1 -> null
```

### Python Solution (Iterative)

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head: ListNode) -> ListNode:
    prev = None
    curr = head

    while curr:
        next_node = curr.next  # Save next
        curr.next = prev       # Reverse pointer
        prev = curr            # Move prev forward
        curr = next_node       # Move curr forward

    return prev  # New head
```

### Python Solution (Recursive)

```python
def reverse_list_recursive(head: ListNode) -> ListNode:
    # Base case: empty or single node
    if not head or not head.next:
        return head

    # Reverse the rest of the list
    new_head = reverse_list_recursive(head.next)

    # head.next is now the tail of reversed list
    # Make it point back to head
    head.next.next = head
    head.next = None

    return new_head
```

### Keywords That Indicated This Solution

- **"Reverse"** → Classic linked list manipulation
- **"In-place"** → Pointer manipulation, no new nodes
- **"Linked list"** → Think about pointer rewiring

### Intuition

**Iterative:** Walk through the list, reversing each pointer as you go. Need three pointers: `prev` (where to point back), `curr` (node being processed), `next` (saved before we overwrite `curr.next`).

**Recursive:** Trust that `reverse_list(head.next)` correctly reverses everything after head. Then just attach head to the end of that reversed list.

**Mental Model (Iterative):** "Take each arrow and flip it. Need to save 'next' before I break the link."

---

## 8. Trees - DFS

### Problem: Lowest Common Ancestor of Binary Tree

**Statement:** Given a binary tree and two nodes `p` and `q`, find their lowest common ancestor (LCA). The LCA is the deepest node that has both p and q as descendants (a node can be a descendant of itself).

```
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
```

### Python Solution

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    # Base case: null or found p or q
    if not root or root == p or root == q:
        return root

    # Search in left and right subtrees
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    # If both sides return non-null, current node is LCA
    if left and right:
        return root

    # Otherwise, return whichever side found something
    return left if left else right
```

### Keywords That Indicated This Solution

- **"Lowest"** / **"deepest"** → Need to explore deep first → DFS
- **"Ancestor"** → Work from bottom up → Post-order DFS
- **"Binary tree"** → Tree traversal
- **"Two nodes"** → Find both, determine where they meet

### Intuition

DFS down the tree. If we find p or q, return it upward. If a node receives non-null from both children, that means p and q are in different subtrees—this node is their meeting point (LCA). If only one side returns non-null, pass that result up.

**Mental Model:** "Send scouts down both paths. If both scouts report back to a node, that node is the meeting point."

---

## 9. Trees - BFS

### Problem: Binary Tree Level Order Traversal

**Statement:** Given a binary tree, return the level order traversal of its nodes' values (i.e., from left to right, level by level).

```
Input: root = [3,9,20,null,null,15,7]
Output: [[3], [9,20], [15,7]]
```

### Python Solution

```python
from collections import deque

def level_order(root: TreeNode) -> list[list[int]]:
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        current_level = []

        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(current_level)

    return result
```

### Keywords That Indicated This Solution

- **"Level order"** / **"level by level"** → BFS
- **"Left to right"** → Queue maintains order
- **"Group by level"** → Track level size before processing

### Intuition

BFS naturally visits nodes level by level. The key trick: before processing a level, record `len(queue)`. That's exactly how many nodes belong to the current level. Process exactly that many, adding their children (next level) to the queue.

**Mental Model:** "Process the queue in batches. Each batch = one level."

---

## 10. Graphs - DFS

### Problem: Number of Islands

**Statement:** Given an `m x n` 2D grid of '1's (land) and '0's (water), return the number of islands. An island is surrounded by water and formed by connecting adjacent lands horizontally or vertically.

```
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
```

### Python Solution

```python
def num_islands(grid: list[list[str]]) -> int:
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    islands = 0

    def dfs(r, c):
        # Boundary and water check
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return

        # Mark as visited (sink the land)
        grid[r][c] = '0'

        # Explore all 4 directions
        dfs(r - 1, c)  # up
        dfs(r + 1, c)  # down
        dfs(r, c - 1)  # left
        dfs(r, c + 1)  # right

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                islands += 1
                dfs(r, c)  # Sink the entire island

    return islands
```

### Keywords That Indicated This Solution

- **"Connected"** / **"adjacent"** → Graph traversal (DFS or BFS)
- **"Number of islands"** / **"connected components"** → Count DFS/BFS starts
- **"Grid"** → 2D graph, neighbors are up/down/left/right
- **"Surrounded by"** → Flood fill pattern

### Intuition

Each '1' we find could be a new island. Start DFS from it to mark all connected land as visited (we "sink" it by changing to '0'). Count how many times we start a new DFS—that's the number of islands.

**Mental Model:** "When I find land, I've discovered an island. Flood-fill to mark the whole island as explored, then keep looking for undiscovered islands."

---

## 11. Graphs - BFS (Shortest Path)

### Problem: Shortest Path in Binary Matrix

**Statement:** Given an `n x n` binary matrix `grid`, return the length of the shortest clear path from top-left to bottom-right. A clear path consists of 0s, and you can move in 8 directions. Return -1 if no path exists.

```
Input: grid = [[0,1],[1,0]]
Output: 2
Path: (0,0) -> (1,1)
```

### Python Solution

```python
from collections import deque

def shortest_path_binary_matrix(grid: list[list[int]]) -> int:
    n = len(grid)

    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1

    # 8 directions
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    queue = deque([(0, 0, 1)])  # row, col, path length
    grid[0][0] = 1  # Mark visited

    while queue:
        row, col, dist = queue.popleft()

        # Reached destination
        if row == n - 1 and col == n - 1:
            return dist

        # Explore all 8 directions
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if (0 <= new_row < n and
                0 <= new_col < n and
                grid[new_row][new_col] == 0):

                grid[new_row][new_col] = 1  # Mark visited
                queue.append((new_row, new_col, dist + 1))

    return -1  # No path found
```

### Keywords That Indicated This Solution

- **"Shortest path"** → BFS (for unweighted graphs)
- **"Minimum steps"** / **"minimum distance"** → BFS
- **"Grid"** + **"path from A to B"** → BFS on grid
- **"Binary matrix"** → 0/1 usually means passable/blocked

### Intuition

BFS explores all cells at distance K before any cell at distance K+1. So the first time we reach the destination, we've found the shortest path. This only works for unweighted graphs (all edges cost 1).

**Mental Model:** "Expand outward like ripples in a pond. The first ripple to reach the goal is the shortest path."

---

## 12. Heap / Priority Queue

### Problem: Merge K Sorted Lists

**Statement:** Given an array of `k` linked lists, each sorted in ascending order, merge all lists into one sorted linked list.

```
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
```

### Python Solution

```python
import heapq

def merge_k_lists(lists: list[ListNode]) -> ListNode:
    # Min heap: (value, index, node)
    # Index is needed to break ties (nodes aren't comparable)
    heap = []

    # Add head of each list to heap
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))

    # Dummy head for result
    dummy = ListNode(0)
    current = dummy

    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next

        # Add next node from same list
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

### Keywords That Indicated This Solution

- **"Merge K sorted"** → Heap to efficiently get minimum among K elements
- **"K sorted arrays/lists"** → Heap of size K
- **"Smallest among multiple sources"** → Min heap
- **"Top K"** / **"Kth smallest"** → Heap

### Intuition

At any moment, the next smallest element must be the head of one of the K lists. Instead of comparing all K heads (O(K) per element), use a min-heap to get the minimum in O(log K). Extract min, add to result, push that list's next element into heap.

**Mental Model:** "K contestants, each with a line of numbers. Keep asking 'who has the smallest front number?' A heap answers this efficiently."

---

## 13. Backtracking

### Problem: Combination Sum

**Statement:** Given an array of distinct integers `candidates` and a target integer `target`, return all unique combinations of candidates where the chosen numbers sum to target. The same number may be chosen unlimited times.

```
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
```

### Python Solution

```python
def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    result = []

    def backtrack(remaining, start, current):
        # Base case: found valid combination
        if remaining == 0:
            result.append(current[:])  # Add copy!
            return

        # Base case: overshot
        if remaining < 0:
            return

        # Try each candidate from 'start' onwards
        for i in range(start, len(candidates)):
            # Choose
            current.append(candidates[i])

            # Explore (start from i, not i+1, to allow reuse)
            backtrack(remaining - candidates[i], i, current)

            # Un-choose (backtrack)
            current.pop()

    backtrack(target, 0, [])
    return result
```

### Keywords That Indicated This Solution

- **"All combinations"** / **"all possibilities"** → Backtracking
- **"Unique combinations"** → Use start index to avoid duplicates
- **"Sum to target"** → Track remaining sum
- **"Unlimited use"** → Start from i (not i+1) in recursion
- **"Subsets"** / **"permutations"** → Backtracking patterns

### Intuition

Backtracking = controlled brute force. At each step, try adding each candidate. If we hit the target, record the combination. If we overshoot, backtrack. The `start` parameter prevents duplicate combinations like [2,3] and [3,2].

**Mental Model:** "Build combinations one choice at a time. If it works, save it. If not, undo the last choice and try another. Never go backwards in the candidate list to avoid duplicates."

---

## 14. Graph - Topological Sort

### Problem: Course Schedule

**Statement:** There are `numCourses` courses labeled 0 to numCourses-1. Given prerequisites where `[a, b]` means you must take course b before course a, determine if it's possible to finish all courses.

```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: True
Explanation: Take course 0, then course 1.

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: False
Explanation: Circular dependency.
```

### Python Solution (BFS - Kahn's Algorithm)

```python
from collections import deque, defaultdict

def can_finish(num_courses: int, prerequisites: list[list[int]]) -> bool:
    # Build adjacency list and indegree count
    graph = defaultdict(list)
    indegree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        indegree[course] += 1

    # Start with courses that have no prerequisites
    queue = deque([i for i in range(num_courses) if indegree[i] == 0])
    completed = 0

    while queue:
        course = queue.popleft()
        completed += 1

        # Reduce indegree of dependent courses
        for next_course in graph[course]:
            indegree[next_course] -= 1
            if indegree[next_course] == 0:
                queue.append(next_course)

    # If we completed all courses, no cycle exists
    return completed == num_courses
```

### Python Solution (DFS - Cycle Detection)

```python
def can_finish_dfs(num_courses: int, prerequisites: list[list[int]]) -> bool:
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    # 0 = unvisited, 1 = visiting, 2 = visited
    state = [0] * num_courses

    def has_cycle(course):
        if state[course] == 1:  # Currently in recursion stack = cycle
            return True
        if state[course] == 2:  # Already fully processed
            return False

        state[course] = 1  # Mark as visiting

        for next_course in graph[course]:
            if has_cycle(next_course):
                return True

        state[course] = 2  # Mark as visited
        return False

    for course in range(num_courses):
        if has_cycle(course):
            return False

    return True
```

### Keywords That Indicated This Solution

- **"Prerequisites"** / **"dependencies"** → Topological sort
- **"Order of tasks"** → Topological sort
- **"Possible to finish"** → Cycle detection in directed graph
- **"Before/after relationship"** → Directed graph, topological order

### Intuition

Model as a directed graph: edge from A to B means "A must come before B". Topological sort finds a valid ordering. If a cycle exists, no valid ordering is possible.

**Kahn's (BFS):** Repeatedly take nodes with no incoming edges (indegree = 0). If we can take all nodes, no cycle exists.

**DFS:** If we revisit a node that's currently in our recursion stack, we found a cycle.

**Mental Model:** "Keep taking courses with no unfulfilled prerequisites. Cross them off, update prerequisites of remaining courses. If we empty the list, we're done. If we get stuck with everyone waiting on someone else, there's a cycle."

---

## Bonus Patterns

### 15. Two Pointers - Fast & Slow (Linked List Cycle)

### Problem: Linked List Cycle

**Statement:** Given head of a linked list, determine if it has a cycle.

```python
def has_cycle(head: ListNode) -> bool:
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            return True

    return False
```

**Keywords:** "cycle", "loop", "circular"

**Intuition:** Fast pointer moves 2x speed. If there's a cycle, fast will eventually lap slow. If no cycle, fast hits null.

---

### 16. Prefix Sum

### Problem: Subarray Sum Equals K

**Statement:** Given an array and integer k, return count of subarrays that sum to k.

```python
def subarray_sum(nums: list[int], k: int) -> int:
    count = 0
    prefix_sum = 0
    seen = {0: 1}  # prefix_sum -> count

    for num in nums:
        prefix_sum += num

        # If (prefix_sum - k) exists, those subarrays sum to k
        if prefix_sum - k in seen:
            count += seen[prefix_sum - k]

        seen[prefix_sum] = seen.get(prefix_sum, 0) + 1

    return count
```

**Keywords:** "subarray sum", "contiguous sum equals"

**Intuition:** If prefix[j] - prefix[i] = k, then subarray [i+1, j] sums to k. Store prefix sums in hashmap.

---

### 17. Binary Search - On Answer

### Problem: Koko Eating Bananas

**Statement:** Koko has `piles` of bananas and `h` hours. Find minimum eating speed `k` (bananas/hour) to finish all piles in time.

```python
import math

def min_eating_speed(piles: list[int], h: int) -> int:
    def can_finish(speed):
        hours = sum(math.ceil(pile / speed) for pile in piles)
        return hours <= h

    left, right = 1, max(piles)

    while left < right:
        mid = (left + right) // 2

        if can_finish(mid):
            right = mid  # Try smaller speed
        else:
            left = mid + 1  # Need faster

    return left
```

**Keywords:** "minimum speed", "minimum capacity", "can finish in time"

**Intuition:** Binary search on the answer space. Check if a given answer works, then narrow the range.

---

## Python Tips for Interviews

### Essential Imports

```python
from collections import defaultdict, deque, Counter
from heapq import heappush, heappop, heapify
from typing import List, Optional
import math
```

### Common Operations Cheat Sheet

```python
# Dictionary
d = {}
d.get(key, default)
d.setdefault(key, default)
d = defaultdict(list)  # Auto-creates empty list for missing keys
d = defaultdict(int)   # Auto-creates 0 for missing keys
Counter(list)          # Frequency count

# Heap (min heap by default)
heappush(heap, val)
heappop(heap)
heap[0]                # Peek
heapify(list)          # Convert list to heap in O(n)
# Max heap: negate values
heappush(heap, -val)

# Deque (double-ended queue)
q = deque()
q.append(x)            # Add right
q.appendleft(x)        # Add left
q.pop()                # Remove right
q.popleft()            # Remove left

# List tricks
list[::-1]             # Reverse
list[-1]               # Last element
list[start:end]        # Slice
list.copy() or list[:] # Shallow copy

# String tricks
''.join(list)          # List to string
s[::-1]                # Reverse string
ord('a')               # Character to ASCII
chr(97)                # ASCII to character

# Sorting
sorted(list)                       # Returns new sorted list
list.sort()                        # In-place sort
sorted(list, key=lambda x: x[1])   # Sort by second element
sorted(list, reverse=True)         # Descending

# Math
float('inf')           # Infinity
float('-inf')          # Negative infinity
math.ceil(x)           # Round up
math.floor(x)          # Round down
divmod(a, b)           # Returns (a // b, a % b)
```

### Time Complexity Reference

| Operation | List   | Dict/Set | Deque | Heap     |
| --------- | ------ | -------- | ----- | -------- |
| Access    | O(1)   | O(1)     | O(n)  | -        |
| Search    | O(n)   | O(1)     | O(n)  | O(n)     |
| Insert    | O(n)\* | O(1)     | O(1)  | O(log n) |
| Delete    | O(n)   | O(1)     | O(1)  | O(log n) |
| Min/Max   | O(n)   | O(n)     | O(n)  | O(1)     |

\*List append is O(1) amortized
