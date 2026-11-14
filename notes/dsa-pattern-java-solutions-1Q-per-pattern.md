# DSA Patterns: Problem Statements, Java Solutions & Intuition

<a id="quick-navigation"></a>

## Quick navigation

- [0. Java Language & Data Structure Cheat Sheet](#section-0-java-language-and-data-structure-cheat-sheet)
- [1. Arrays & Hashing](#section-1-arrays-and-hashing)
- [2. Two Pointers](#section-2-two-pointers)
- [3. Sliding Window](#section-3-sliding-window)
- [4. Binary Search](#section-4-binary-search)
- [5. Stack](#section-5-stack)
- [6. Queue / BFS](#section-6-queue-bfs)
- [7. Linked List](#section-7-linked-list)
- [8. Trees - DFS](#section-8-trees-dfs)
- [9. Trees - BFS](#section-9-trees-bfs)
- [10. Graphs - DFS](#section-10-graphs-dfs)
- [11. Graphs - BFS (Shortest Path)](#section-11-graphs-bfs-shortest-path)
- [12. Heap / Priority Queue](#section-12-heap-priority-queue)
- [13. Backtracking](#section-13-backtracking)
- [14. Graph - Topological Sort](#section-14-graph-topological-sort)
- [Graph Mastery Expansion: One Question Per Advanced Pattern](#section-graph-mastery-expansion-one-question-per-advanced-pattern)
- [Python vs Java for Interviews](#section-python-vs-java-for-interviews)


---

<a id="section-0-java-language-and-data-structure-cheat-sheet"></a>

## 0. Java Language & Data Structure Cheat Sheet

### Imports, methods, and variables

```java
import java.util.*;

public int solve(int[] nums) {
    int answer = 0;
    return answer;
}

int count = 0;
long total = 0L;               // use long for sums/distances that may overflow int
double ratio = 0.0;
boolean found = false;
char letter = 'a';
String text = "graph";

int temporary = left;          // swap
left = right;
right = temporary;
```

### Loops

```java
for (int index = 0; index < nums.length; index++) {
    int value = nums[index];
}

for (int index = nums.length - 1; index >= 0; index--) {
    int value = nums[index];
}

for (int value : nums) {       // enhanced for: values, not indices
}

for (int row = 0; row < grid.length; row++) {
    for (int col = 0; col < grid[0].length; col++) {
    }
}

for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
    int key = entry.getKey();
    int value = entry.getValue();
}

Iterator<Integer> iterator = set.iterator();
while (iterator.hasNext()) {
    int value = iterator.next();
    // iterator.remove();        // safe removal during this iteration
}

while (left <= right) {
    left++;
}
// break exits one loop; continue skips to its next iteration.
```

### Primitive arrays and matrices

```java
int[] nums = {3, 1, 2};
int[] zeros = new int[n];
int[][] matrix = new int[rows][cols];

int length = nums.length;
int value = nums[index];
nums[index] = 99;

Arrays.fill(nums, -1);
Arrays.sort(nums);                         // ascending, mutates array
int[] copy = Arrays.copyOf(nums, nums.length);
int[] slice = Arrays.copyOfRange(nums, from, to); // to excluded

int position = Arrays.binarySearch(nums, target); // sorted input; negative if absent
boolean exists = position >= 0;

for (int[] row : matrix) Arrays.fill(row, -1);
```

Java arrays have fixed length: you cannot append or remove. Use `ArrayList` when the size changes.

### ArrayList / dynamic array

```java
List<Integer> list = new ArrayList<>();
list.add(value);                    // append
list.add(index, value);             // insert at index: O(n)
list.addAll(other);

int value = list.get(index);
list.set(index, newValue);          // replace
int removed = list.remove(index);   // remove by index
boolean didRemove = list.remove(Integer.valueOf(target)); // remove by value
boolean exists = list.contains(target);
int position = list.indexOf(target);

int size = list.size();
boolean empty = list.isEmpty();
list.clear();

list.sort(Integer::compareTo);
list.sort(Comparator.reverseOrder());
pairs.sort(Comparator.comparingInt(pair -> pair[1]));

List<Integer> copy = new ArrayList<>(list);
List<Integer> view = list.subList(from, to); // backed by original list
```

**Classic trap:** `list.remove(2)` removes index 2, not the value 2. Use `remove(Integer.valueOf(2))`.

### Strings and StringBuilder

```java
String text = "graph";
int length = text.length();
char character = text.charAt(index);
String substring = text.substring(start, end); // end excluded
boolean contains = text.contains("rap");
boolean starts = text.startsWith("gr");
int position = text.indexOf('a');              // -1 if absent
String[] parts = text.split(",");
String joined = String.join("-", parts);
String trimmed = text.trim();
String replaced = text.replace('a', 'b');

boolean sameContent = first.equals(second);     // never use == for content
int ordering = first.compareTo(second);
char[] characters = text.toCharArray();
String fromCharacters = new String(characters);
int number = Integer.parseInt("42");
String digits = String.valueOf(42);

StringBuilder builder = new StringBuilder();
builder.append("ab");
builder.append(value);
builder.insert(index, value);
builder.setCharAt(index, 'X');
builder.deleteCharAt(index);
builder.delete(start, end);
builder.reverse();
String result = builder.toString();
```

Strings are immutable. Use `StringBuilder` for repeated edits or concatenation inside loops.

### HashMap

```java
Map<Integer, Integer> map = new HashMap<>();
map.put(key, value);                       // add or replace
map.putIfAbsent(key, value);

int value = map.get(key);                  // may unbox null and throw
int safeValue = map.getOrDefault(key, 0);
boolean hasKey = map.containsKey(key);
boolean hasValue = map.containsValue(value); // O(n)

map.merge(key, 1, Integer::sum);            // frequency increment
Map<Integer, List<Integer>> groups = new HashMap<>();
groups.computeIfAbsent(key, ignored -> new ArrayList<>()).add(value);

Integer removed = map.remove(key);
boolean removedPair = map.remove(key, value);
int size = map.size();
map.clear();

for (int key : map.keySet()) {
    int mappedValue = map.get(key);
}
for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
    // entry.getKey(), entry.getValue()
}
```

Average `HashMap` lookup/add/remove is O(1). Use `TreeMap` when keys must stay sorted: operations become O(log n).

### HashSet

```java
Set<Integer> seen = new HashSet<>();
boolean added = seen.add(value);
boolean exists = seen.contains(value);
boolean removed = seen.remove(value);
seen.addAll(values);
seen.removeAll(values);
seen.retainAll(values);             // intersection, mutates seen
int size = seen.size();
boolean empty = seen.isEmpty();
seen.clear();

Set<Integer> ordered = new TreeSet<>(); // sorted, O(log n) operations
```

### Stack, queue, and deque

```java
// Stack: prefer Deque over the legacy Stack class
Deque<Integer> stack = new ArrayDeque<>();
stack.push(value);
int top = stack.peek();
int stackRemoved = stack.pop();
boolean stackEmpty = stack.isEmpty();

// Queue
Queue<Integer> queue = new ArrayDeque<>();
queue.offer(value);                  // enqueue
int front = queue.peek();
int queueRemoved = queue.poll();     // dequeue
boolean queueEmpty = queue.isEmpty();

// Double-ended queue
Deque<Integer> deque = new ArrayDeque<>();
deque.offerFirst(value);
deque.offerLast(value);
int first = deque.peekFirst();
int last = deque.peekLast();
int removedFirst = deque.pollFirst();
int removedLast = deque.pollLast();
```

The `offer/poll/peek` family returns a status or `null` rather than throwing on ordinary queue failure. `ArrayDeque` does not allow `null`.

### Heap / PriorityQueue

```java
PriorityQueue<Integer> minHeap = new PriorityQueue<>();
minHeap.offer(value);
int smallest = minHeap.peek();
int removed = minHeap.poll();

PriorityQueue<Integer> maxHeap =
    new PriorityQueue<>(Comparator.reverseOrder());

PriorityQueue<int[]> byDistance =
    new PriorityQueue<>(Comparator.comparingInt(item -> item[0]));
byDistance.offer(new int[]{distance, node});
int[] next = byDistance.poll();

PriorityQueue<long[]> safeLongComparator =
    new PriorityQueue<>(Comparator.comparingLong(item -> item[0]));
```

Push/pop is O(log n), peek is O(1). Prefer `Integer.compare(a, b)` or comparator helpers over `a - b`, which can overflow.

### Linked structures and graphs

```java
class ListNode {
    int val;
    ListNode next;
    ListNode(int val) { this.val = val; }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int val) { this.val = val; }
}

// Unweighted adjacency list
List<Integer>[] graph = new ArrayList[n];
for (int node = 0; node < n; node++) graph[node] = new ArrayList<>();
for (int[] edge : edges) {
    int first = edge[0], second = edge[1];
    graph[first].add(second);
    graph[second].add(first);         // omit for a directed graph
}

// Weighted: each entry is {neighbour, weight}
List<int[]>[] weighted = new ArrayList[n];
for (int node = 0; node < n; node++) weighted[node] = new ArrayList<>();
weighted[source].add(new int[]{target, weight});

boolean[] visited = new boolean[n];
int[][] directions4 = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
```

### Conversions and interview traps

```java
int[] primitive = list.stream().mapToInt(Integer::intValue).toArray();
List<Integer> boxed = new ArrayList<>();
for (int value : nums) boxed.add(value);

int infinity = Integer.MAX_VALUE;
long safeInfinity = Long.MAX_VALUE / 4; // leaves room for addition
```

- Use `.length` for arrays, `.length()` for strings, and `.size()` for collections.
- Generics cannot use primitives: write `List<Integer>`, not `List<int>`.
- Check for `null` before unboxing values returned from maps or queues.
- Modifying a collection inside enhanced `for` usually throws `ConcurrentModificationException`; use an iterator or collect changes separately.
- Recursion can overflow the JVM stack on deep graphs; use iterative DFS when input depth is large.

### Complexity reminders

| Structure | Add | Remove | Membership / lookup | Peek |
| --------- | --- | ------ | ------------------- | ---- |
| array | — | — | O(n), index O(1) | O(1) |
| `ArrayList` end | O(1)* | O(1) | O(n), index O(1) | O(1) |
| `HashMap` / `HashSet` average | O(1) | O(1) | O(1) | — |
| `TreeMap` / `TreeSet` | O(log n) | O(log n) | O(log n) | O(log n) |
| `ArrayDeque` ends | O(1) | O(1) | O(n) | O(1) |
| `PriorityQueue` | O(log n) | O(log n) | O(n) | O(1) |

*Amortised.

---


<a id="section-1-arrays-and-hashing"></a>

## 1. Arrays & Hashing

### Problem: Two Sum

**Statement:** Given an array of integers `nums` and an integer `target`, return the indices of the two numbers such that they add up to target. You may assume each input has exactly one solution.

```
Input: nums = [2, 7, 11, 15], target = 9
Output: [0, 1]
Explanation: nums[0] + nums[1] = 2 + 7 = 9
```

### Java Solution

```java
import java.util.HashMap;
import java.util.Map;

public class TwoSum {
    public int[] twoSum(int[] nums, int target) {
        // Map to store: value -> index
        Map<Integer, Integer> seen = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];

            // Check if complement was seen before
            if (seen.containsKey(complement)) {
                return new int[] { seen.get(complement), i };
            }

            // Store current number and its index
            seen.put(nums[i], i);
        }

        return new int[] {}; // No solution found
    }
}
```

### Keywords That Indicated This Solution

- **"Two numbers that add up to"** → Need to find a pair → HashMap for O(1) lookup
- **"Return indices"** → Store index in HashMap, not just existence
- **"Exactly one solution"** → Don't need to handle duplicates/multiple answers

### Intuition

The brute force is O(n²): check every pair. But notice: if we need `a + b = target`, then when we see `a`, we're really looking for `target - a`. Instead of scanning the whole array for the complement, we store everything we've seen in a HashMap. When we reach element `b`, we check if `target - b` exists in our map. This reduces O(n²) → O(n).

**Mental Model:** "I've seen these numbers before, is my partner already waiting for me?"

---

<a id="section-2-two-pointers"></a>

## 2. Two Pointers

### Problem: Container With Most Water

**Statement:** Given an array `height` where `height[i]` is the height of a vertical line at position `i`, find two lines that together with the x-axis forms a container that holds the most water.

```
Input: height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
Output: 49
Explanation: Lines at index 1 (height 8) and index 8 (height 7) form container with area = 7 * 7 = 49
```

### Java Solution

```java
public class ContainerWithMostWater {
    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int maxWater = 0;

        while (left < right) {
            // Calculate current area
            int width = right - left;
            int h = Math.min(height[left], height[right]);
            int area = width * h;
            maxWater = Math.max(maxWater, area);

            // Move the pointer with smaller height
            // (moving the taller one can never increase area)
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }

        return maxWater;
    }
}
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

<a id="section-3-sliding-window"></a>

## 3. Sliding Window

### Problem: Longest Substring Without Repeating Characters

**Statement:** Given a string `s`, find the length of the longest substring without repeating characters.

```
Input: s = "abcabcbb"
Output: 3
Explanation: "abc" is the longest substring without repeating characters.
```

### Java Solution

```java
import java.util.HashSet;
import java.util.Set;

public class LongestSubstringWithoutRepeating {
    public int lengthOfLongestSubstring(String s) {
        Set<Character> window = new HashSet<>();
        int left = 0;
        int maxLength = 0;

        for (int right = 0; right < s.length(); right++) {
            char c = s.charAt(right);

            // Shrink window until no duplicate
            while (window.contains(c)) {
                window.remove(s.charAt(left));
                left++;
            }

            // Add current character to window
            window.add(c);

            // Update max length
            maxLength = Math.max(maxLength, right - left + 1);
        }

        return maxLength;
    }
}
```

### Alternative: HashMap to Jump Directly

```java
import java.util.HashMap;
import java.util.Map;

public class LongestSubstringOptimized {
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> lastSeen = new HashMap<>();
        int left = 0;
        int maxLength = 0;

        for (int right = 0; right < s.length(); right++) {
            char c = s.charAt(right);

            // If character seen and is within current window, jump left pointer
            if (lastSeen.containsKey(c) && lastSeen.get(c) >= left) {
                left = lastSeen.get(c) + 1;
            }

            lastSeen.put(c, right);
            maxLength = Math.max(maxLength, right - left + 1);
        }

        return maxLength;
    }
}
```

### Keywords That Indicated This Solution

- **"Substring"** → Contiguous → Sliding window
- **"Longest"** → Variable-size window, expand and track max
- **"Without repeating"** → Constraint that determines when to shrink

### Intuition

We want contiguous characters (substring), so sliding window fits. Expand the right pointer to include more characters. When we hit a duplicate, the window is invalid—shrink from the left until valid again. Track the maximum valid window size seen.

**Mental Model:** "Grow my window greedily. When it breaks the rule, shrink from left until it's valid again."

---

<a id="section-4-binary-search"></a>

## 4. Binary Search

### Problem: Search in Rotated Sorted Array

**Statement:** Given a rotated sorted array `nums` and a `target`, return the index of target or -1 if not found. The array was originally sorted in ascending order, then rotated at some pivot.

```
Input: nums = [4, 5, 6, 7, 0, 1, 2], target = 0
Output: 4
```

### Java Solution

```java
public class SearchRotatedArray {
    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] == target) {
                return mid;
            }

            // Determine which half is sorted
            if (nums[left] <= nums[mid]) {
                // Left half is sorted
                if (target >= nums[left] && target < nums[mid]) {
                    // Target is in the sorted left half
                    right = mid - 1;
                } else {
                    // Target is in the right half
                    left = mid + 1;
                }
            } else {
                // Right half is sorted
                if (target > nums[mid] && target <= nums[right]) {
                    // Target is in the sorted right half
                    left = mid + 1;
                } else {
                    // Target is in the left half
                    right = mid - 1;
                }
            }
        }

        return -1;
    }
}
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

<a id="section-5-stack"></a>

## 5. Stack

### Problem: Daily Temperatures

**Statement:** Given an array `temperatures`, return an array `answer` where `answer[i]` is the number of days you have to wait after the i-th day to get a warmer temperature. If no future day is warmer, put 0.

```
Input: temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
Output: [1, 1, 4, 2, 1, 1, 0, 0]
```

### Java Solution

```java
import java.util.Stack;

public class DailyTemperatures {
    public int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        int[] answer = new int[n];
        Stack<Integer> stack = new Stack<>();  // Store indices

        for (int i = 0; i < n; i++) {
            // Pop all days that found their warmer day (today)
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                int prevDay = stack.pop();
                answer[prevDay] = i - prevDay;
            }

            // Push current day (waiting for a warmer day)
            stack.push(i);
        }

        // Days left in stack have no warmer day (answer stays 0)
        return answer;
    }
}
```

### Keywords That Indicated This Solution

- **"Next greater"** / **"next warmer"** → Classic monotonic stack pattern
- **"Wait until"** / **"how many days until"** → Looking forward, stack stores pending items
- **"For each element, find the next X"** → Stack to track unresolved elements

### Intuition

Each day is "waiting" for a warmer day. Push days onto the stack. When a warmer day comes, it resolves all the cooler days waiting on the stack. Pop them and record how long they waited. The stack maintains a decreasing order of temperatures (monotonic decreasing stack).

**Mental Model:** "Each day goes into a waiting room (stack). When a hotter day arrives, it 'answers' everyone in the waiting room who was cooler."

---

<a id="section-6-queue-bfs"></a>

## 6. Queue / BFS

### Problem: Rotting Oranges

**Statement:** In a grid, 0 = empty, 1 = fresh orange, 2 = rotten orange. Every minute, fresh oranges adjacent to rotten ones become rotten. Return the minimum minutes until no fresh orange remains, or -1 if impossible.

```
Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4
```

### Java Solution

```java
import java.util.LinkedList;
import java.util.Queue;

public class RottingOranges {
    public int orangesRotting(int[][] grid) {
        int rows = grid.length;
        int cols = grid[0].length;
        Queue<int[]> queue = new LinkedList<>();
        int freshCount = 0;

        // Initialize: find all rotten oranges and count fresh ones
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (grid[r][c] == 2) {
                    queue.offer(new int[] {r, c});
                } else if (grid[r][c] == 1) {
                    freshCount++;
                }
            }
        }

        if (freshCount == 0) return 0;

        int minutes = 0;
        int[] deltaRow = {-1, 0, 1, 0};
        int[] deltaCol = {0, 1, 0, -1};

        // BFS level by level
        while (!queue.isEmpty()) {
            int size = queue.size();
            boolean rottenThisMinute = false;

            for (int i = 0; i < size; i++) {
                int[] cell = queue.poll();
                int row = cell[0];
                int col = cell[1];

                // Check all 4 neighbors
                for (int d = 0; d < 4; d++) {
                    int newRow = row + deltaRow[d];
                    int newCol = col + deltaCol[d];

                    if (newRow >= 0 && newRow < rows &&
                        newCol >= 0 && newCol < cols &&
                        grid[newRow][newCol] == 1) {

                        grid[newRow][newCol] = 2;  // Mark rotten
                        queue.offer(new int[] {newRow, newCol});
                        freshCount--;
                        rottenThisMinute = true;
                    }
                }
            }

            if (rottenThisMinute) {
                minutes++;
            }
        }

        return freshCount == 0 ? minutes : -1;
    }
}
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

<a id="section-7-linked-list"></a>

## 7. Linked List

### Problem: Reverse Linked List

**Statement:** Given the head of a singly linked list, reverse the list and return the new head.

```
Input: 1 -> 2 -> 3 -> 4 -> 5 -> null
Output: 5 -> 4 -> 3 -> 2 -> 1 -> null
```

### Java Solution (Iterative)

```java
public class ReverseLinkedList {
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;

        while (curr != null) {
            ListNode next = curr.next;  // Save next
            curr.next = prev;           // Reverse pointer
            prev = curr;                // Move prev forward
            curr = next;                // Move curr forward
        }

        return prev;  // New head
    }
}

class ListNode {
    int val;
    ListNode next;
    ListNode(int val) { this.val = val; }
}
```

### Java Solution (Recursive)

```java
public class ReverseLinkedListRecursive {
    public ListNode reverseList(ListNode head) {
        // Base case: empty or single node
        if (head == null || head.next == null) {
            return head;
        }

        // Reverse the rest of the list
        ListNode newHead = reverseList(head.next);

        // head.next is now the last node of reversed list
        // Make it point back to head
        head.next.next = head;
        head.next = null;

        return newHead;
    }
}
```

### Keywords That Indicated This Solution

- **"Reverse"** → Classic linked list manipulation
- **"In-place"** → Pointer manipulation, no new nodes
- **"Linked list"** → Think about pointer rewiring

### Intuition

**Iterative:** Walk through the list, reversing each pointer as you go. Need three pointers: `prev` (where to point back), `curr` (node being processed), `next` (saved before we overwrite `curr.next`).

**Recursive:** Trust that `reverseList(head.next)` correctly reverses everything after head. Then just attach head to the end of that reversed list.

**Mental Model (Iterative):** "Take each arrow and flip it. Need to save 'next' before I break the link."

---

<a id="section-8-trees-dfs"></a>

## 8. Trees - DFS

### Problem: Lowest Common Ancestor of Binary Tree

**Statement:** Given a binary tree and two nodes `p` and `q`, find their lowest common ancestor (LCA). The LCA is the deepest node that has both p and q as descendants (a node can be a descendant of itself).

```
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
```

### Java Solution

```java
public class LowestCommonAncestor {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // Base case: null or found p or q
        if (root == null || root == p || root == q) {
            return root;
        }

        // Search in left and right subtrees
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);

        // If both sides return non-null, current node is LCA
        if (left != null && right != null) {
            return root;
        }

        // Otherwise, return whichever side found something
        return left != null ? left : right;
    }
}

class TreeNode {
    int val;
    TreeNode left, right;
    TreeNode(int val) { this.val = val; }
}
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

<a id="section-9-trees-bfs"></a>

## 9. Trees - BFS

### Problem: Binary Tree Level Order Traversal

**Statement:** Given a binary tree, return the level order traversal of its nodes' values (i.e., from left to right, level by level).

```
Input: root = [3,9,20,null,null,15,7]
Output: [[3], [9,20], [15,7]]
```

### Java Solution

```java
import java.util.*;

public class LevelOrderTraversal {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            int levelSize = queue.size();  // Nodes at current level
            List<Integer> currentLevel = new ArrayList<>();

            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                currentLevel.add(node.val);

                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }

            result.add(currentLevel);
        }

        return result;
    }
}
```

### Keywords That Indicated This Solution

- **"Level order"** / **"level by level"** → BFS
- **"Left to right"** → Queue maintains order
- **"Group by level"** → Track level size before processing

### Intuition

BFS naturally visits nodes level by level. The key trick: before processing a level, record `queue.size()`. That's exactly how many nodes belong to the current level. Process exactly that many, adding their children (next level) to the queue.

**Mental Model:** "Process the queue in batches. Each batch = one level."

---

<a id="section-10-graphs-dfs"></a>

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

### Java Solution

```java
public class NumberOfIslands {
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) return 0;

        int rows = grid.length;
        int cols = grid[0].length;
        int islands = 0;

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (grid[r][c] == '1') {
                    islands++;
                    dfs(grid, r, c);  // Sink the entire island
                }
            }
        }

        return islands;
    }

    private void dfs(char[][] grid, int r, int c) {
        // Boundary and water check
        if (r < 0 || r >= grid.length ||
            c < 0 || c >= grid[0].length ||
            grid[r][c] == '0') {
            return;
        }

        // Mark as visited (sink the land)
        grid[r][c] = '0';

        // Explore all 4 directions
        dfs(grid, r - 1, c);  // up
        dfs(grid, r + 1, c);  // down
        dfs(grid, r, c - 1);  // left
        dfs(grid, r, c + 1);  // right
    }
}
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

<a id="section-11-graphs-bfs-shortest-path"></a>

## 11. Graphs - BFS (Shortest Path)

### Problem: Shortest Path in Binary Matrix

**Statement:** Given an `n x n` binary matrix `grid`, return the length of the shortest clear path from top-left to bottom-right. A clear path consists of 0s, and you can move in 8 directions. Return -1 if no path exists.

```
Input: grid = [[0,1],[1,0]]
Output: 2
Path: (0,0) -> (1,1)
```

### Java Solution

```java
import java.util.*;

public class ShortestPathBinaryMatrix {
    public int shortestPathBinaryMatrix(int[][] grid) {
        int n = grid.length;
        if (grid[0][0] == 1 || grid[n-1][n-1] == 1) return -1;

        // 8 directions
        int[][] directions = {
            {-1,-1}, {-1,0}, {-1,1},
            {0,-1},          {0,1},
            {1,-1},  {1,0},  {1,1}
        };

        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[] {0, 0, 1});  // row, col, path length
        grid[0][0] = 1;  // Mark visited

        while (!queue.isEmpty()) {
            int[] cell = queue.poll();
            int row = cell[0], col = cell[1], dist = cell[2];

            // Reached destination
            if (row == n - 1 && col == n - 1) {
                return dist;
            }

            // Explore all 8 directions
            for (int[] dir : directions) {
                int newRow = row + dir[0];
                int newCol = col + dir[1];

                if (newRow >= 0 && newRow < n &&
                    newCol >= 0 && newCol < n &&
                    grid[newRow][newCol] == 0) {

                    grid[newRow][newCol] = 1;  // Mark visited
                    queue.offer(new int[] {newRow, newCol, dist + 1});
                }
            }
        }

        return -1;  // No path found
    }
}
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

<a id="section-12-heap-priority-queue"></a>

## 12. Heap / Priority Queue

### Problem: Merge K Sorted Lists

**Statement:** Given an array of `k` linked lists, each sorted in ascending order, merge all lists into one sorted linked list.

```
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
```

### Java Solution

```java
import java.util.PriorityQueue;

public class MergeKSortedLists {
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;

        // Min heap ordered by node value
        PriorityQueue<ListNode> minHeap = new PriorityQueue<>(
            (a, b) -> a.val - b.val
        );

        // Add head of each list to heap
        for (ListNode head : lists) {
            if (head != null) {
                minHeap.offer(head);
            }
        }

        // Dummy head for result
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;

        while (!minHeap.isEmpty()) {
            // Get smallest node
            ListNode smallest = minHeap.poll();
            current.next = smallest;
            current = current.next;

            // Add next node from same list
            if (smallest.next != null) {
                minHeap.offer(smallest.next);
            }
        }

        return dummy.next;
    }
}
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

<a id="section-13-backtracking"></a>

## 13. Backtracking

### Problem: Combination Sum

**Statement:** Given an array of distinct integers `candidates` and a target integer `target`, return all unique combinations of candidates where the chosen numbers sum to target. The same number may be chosen unlimited times.

```
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
```

### Java Solution

```java
import java.util.*;

public class CombinationSum {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        backtrack(candidates, target, 0, new ArrayList<>(), result);
        return result;
    }

    private void backtrack(int[] candidates, int remaining, int start,
                           List<Integer> current, List<List<Integer>> result) {
        // Base case: found valid combination
        if (remaining == 0) {
            result.add(new ArrayList<>(current));  // Add copy!
            return;
        }

        // Base case: overshot
        if (remaining < 0) {
            return;
        }

        // Try each candidate from 'start' onwards
        for (int i = start; i < candidates.length; i++) {
            // Choose
            current.add(candidates[i]);

            // Explore (start from i, not i+1, to allow reuse)
            backtrack(candidates, remaining - candidates[i], i, current, result);

            // Un-choose (backtrack)
            current.remove(current.size() - 1);
        }
    }
}
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

<a id="section-14-graph-topological-sort"></a>

## 14. Graph - Topological Sort

### Problem: Course Schedule

**Statement:** There are `numCourses` courses labeled 0 to numCourses-1. Given prerequisites where `[a, b]` means you must take course b before course a, determine if it's possible to finish all courses.

```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: Take course 0, then course 1.

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: Circular dependency.
```

### Java Solution (BFS - Kahn's Algorithm)

```java
import java.util.*;

public class CourseSchedule {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        // Build adjacency list and indegree array
        List<List<Integer>> graph = new ArrayList<>();
        int[] indegree = new int[numCourses];

        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
        }

        for (int[] prereq : prerequisites) {
            int course = prereq[0];
            int prereqCourse = prereq[1];
            graph.get(prereqCourse).add(course);
            indegree[course]++;
        }

        // Start with courses that have no prerequisites
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) {
                queue.offer(i);
            }
        }

        int completed = 0;

        while (!queue.isEmpty()) {
            int course = queue.poll();
            completed++;

            // Reduce indegree of dependent courses
            for (int nextCourse : graph.get(course)) {
                indegree[nextCourse]--;
                if (indegree[nextCourse] == 0) {
                    queue.offer(nextCourse);
                }
            }
        }

        // If we completed all courses, no cycle exists
        return completed == numCourses;
    }
}
```

### Keywords That Indicated This Solution

- **"Prerequisites"** / **"dependencies"** → Topological sort
- **"Order of tasks"** → Topological sort
- **"Possible to finish"** → Cycle detection in directed graph
- **"Before/after relationship"** → Directed graph, topological order

### Intuition

Model as a directed graph: edge from A to B means "A must come before B". Topological sort finds a valid ordering. If a cycle exists, no valid ordering is possible. Kahn's algorithm: repeatedly take nodes with no incoming edges (indegree = 0). If we can take all nodes, no cycle exists.

**Mental Model:** "Keep taking courses with no unfulfilled prerequisites. Cross them off, update prerequisites of remaining courses. If we empty the list, we're done. If we get stuck with everyone waiting on someone else, there's a cycle."

---

<a id="section-graph-mastery-expansion-one-question-per-advanced-pattern"></a>

# Graph Mastery Expansion: One Question Per Advanced Pattern

## G1. Dijkstra - Network Delay Time

**Use when:** shortest paths have non-negative, unequal weights.

```java
public int networkDelayTime(int[][] times, int n, int start) {
    List<int[]>[] graph = new ArrayList[n + 1];
    for (int node = 1; node <= n; node++) graph[node] = new ArrayList<>();
    for (int[] edge : times) graph[edge[0]].add(new int[]{edge[1], edge[2]});

    long[] distance = new long[n + 1];
    Arrays.fill(distance, Long.MAX_VALUE);
    distance[start] = 0;
    PriorityQueue<long[]> minHeap =
        new PriorityQueue<>(Comparator.comparingLong(item -> item[0]));
    minHeap.offer(new long[]{0, start});

    while (!minHeap.isEmpty()) {
        long[] current = minHeap.poll();
        long currentDistance = current[0];
        int node = (int) current[1];
        if (currentDistance != distance[node]) continue;

        for (int[] edge : graph[node]) {
            int neighbour = edge[0];
            long candidate = currentDistance + edge[1];
            if (candidate < distance[neighbour]) {
                distance[neighbour] = candidate;
                minHeap.offer(new long[]{candidate, neighbour});
            }
        }
    }

    long answer = 0;
    for (int node = 1; node <= n; node++) answer = Math.max(answer, distance[node]);
    return answer == Long.MAX_VALUE ? -1 : (int) answer;
}
```

**Invariant:** with non-negative weights, removing the current minimum distance finalises it. Skip stale heap entries rather than marking a node when inserted.

**Complexity:** O((V + E) log V) time, O(V + E) space.

---

## G2. 0-1 BFS - Minimum Obstacle Removal

**Use when:** every edge weight is exactly 0 or 1.

```java
public int minimumObstacles(int[][] grid) {
    int rows = grid.length, cols = grid[0].length;
    int[][] distance = new int[rows][cols];
    for (int[] row : distance) Arrays.fill(row, Integer.MAX_VALUE);
    distance[0][0] = 0;
    Deque<int[]> deque = new ArrayDeque<>();
    deque.offerFirst(new int[]{0, 0});
    int[][] directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    while (!deque.isEmpty()) {
        int[] current = deque.pollFirst();
        int row = current[0], col = current[1];
        for (int[] direction : directions) {
            int nextRow = row + direction[0], nextCol = col + direction[1];
            if (nextRow < 0 || nextRow >= rows || nextCol < 0 || nextCol >= cols) continue;
            int weight = grid[nextRow][nextCol];
            int candidate = distance[row][col] + weight;
            if (candidate >= distance[nextRow][nextCol]) continue;
            distance[nextRow][nextCol] = candidate;
            if (weight == 0) deque.offerFirst(new int[]{nextRow, nextCol});
            else deque.offerLast(new int[]{nextRow, nextCol});
        }
    }
    return distance[rows - 1][cols - 1];
}
```

Free moves go to the front and cost-one moves to the back, preserving shortest-distance order in O(V + E).

---

## G3. Union-Find - Redundant Connection

**Use when:** edges arrive incrementally and you repeatedly merge or query components.

```java
class UnionFind {
    private final int[] parent;
    private final int[] size;

    UnionFind(int n) {
        parent = new int[n];
        size = new int[n];
        for (int node = 0; node < n; node++) {
            parent[node] = node;
            size[node] = 1;
        }
    }

    int find(int node) {
        if (parent[node] != node) parent[node] = find(parent[node]);
        return parent[node];
    }

    boolean union(int first, int second) {
        int rootFirst = find(first), rootSecond = find(second);
        if (rootFirst == rootSecond) return false;
        if (size[rootFirst] < size[rootSecond]) {
            int temporary = rootFirst;
            rootFirst = rootSecond;
            rootSecond = temporary;
        }
        parent[rootSecond] = rootFirst;
        size[rootFirst] += size[rootSecond];
        return true;
    }
}

public int[] findRedundantConnection(int[][] edges) {
    UnionFind unionFind = new UnionFind(edges.length + 1);
    for (int[] edge : edges) {
        if (!unionFind.union(edge[0], edge[1])) return edge;
    }
    return new int[0];
}
```

An undirected edge closes a cycle exactly when its endpoints already have the same representative. Complexity is O(E alpha(V)), effectively linear.

---

## G4. Minimum Spanning Tree - Min Cost to Connect All Points

**Use when:** connect every node while minimising the **total** selected edge cost.

```java
public int minCostConnectPoints(int[][] points) {
    int n = points.length;
    boolean[] inTree = new boolean[n];
    PriorityQueue<int[]> minHeap =
        new PriorityQueue<>(Comparator.comparingInt(edge -> edge[0]));
    minHeap.offer(new int[]{0, 0});
    int totalCost = 0, used = 0;

    while (used < n) {
        int[] edge = minHeap.poll();
        int cost = edge[0], point = edge[1];
        if (inTree[point]) continue;
        inTree[point] = true;
        totalCost += cost;
        used++;

        for (int neighbour = 0; neighbour < n; neighbour++) {
            if (!inTree[neighbour]) {
                int distance = Math.abs(points[point][0] - points[neighbour][0])
                    + Math.abs(points[point][1] - points[neighbour][1]);
                minHeap.offer(new int[]{distance, neighbour});
            }
        }
    }
    return totalCost;
}
```

**Key distinction:** an MST minimises total network cost; Dijkstra minimises path costs from one source.

---

## G5. Tarjan Low-Link - Critical Connections

**Use when:** removing one undirected edge may disconnect the network.

```java
class BridgeFinder {
    private List<int[]>[] graph;
    private int[] discovery;
    private int[] low;
    private int timer;
    private final List<List<Integer>> bridges = new ArrayList<>();

    public List<List<Integer>> criticalConnections(
        int n, List<List<Integer>> connections
    ) {
        bridges.clear();
        timer = 0;
        graph = new ArrayList[n];
        for (int node = 0; node < n; node++) graph[node] = new ArrayList<>();
        for (int edgeId = 0; edgeId < connections.size(); edgeId++) {
            int first = connections.get(edgeId).get(0);
            int second = connections.get(edgeId).get(1);
            graph[first].add(new int[]{second, edgeId});
            graph[second].add(new int[]{first, edgeId});
        }
        discovery = new int[n];
        Arrays.fill(discovery, -1);
        low = new int[n];

        for (int node = 0; node < n; node++) {
            if (discovery[node] == -1) dfs(node, -1);
        }
        return bridges;
    }

    private void dfs(int node, int parentEdge) {
        discovery[node] = low[node] = timer++;
        for (int[] edge : graph[node]) {
            int neighbour = edge[0], edgeId = edge[1];
            if (edgeId == parentEdge) continue;
            if (discovery[neighbour] == -1) {
                dfs(neighbour, edgeId);
                low[node] = Math.min(low[node], low[neighbour]);
                if (low[neighbour] > discovery[node]) {
                    bridges.add(Arrays.asList(node, neighbour));
                }
            } else {
                low[node] = Math.min(low[node], discovery[neighbour]);
            }
        }
    }
}
```

`low[v] > discovery[u]` means the subtree at `v` cannot bypass edge `u-v`. Parent edge IDs keep parallel edges correct.

---

## G6. State-Space BFS - Shortest Path to Get All Keys

**Use when:** location alone does not determine the legal next moves.

```java
public int shortestPathAllKeys(String[] grid) {
    int rows = grid.length, cols = grid[0].length();
    int startRow = 0, startCol = 0, allKeys = 0;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            char cell = grid[row].charAt(col);
            if (cell == '@') { startRow = row; startCol = col; }
            else if (cell >= 'a' && cell <= 'f') allKeys |= 1 << (cell - 'a');
        }
    }

    Queue<int[]> queue = new ArrayDeque<>();
    queue.offer(new int[]{startRow, startCol, 0});
    boolean[][][] visited = new boolean[rows][cols][64];
    visited[startRow][startCol][0] = true;
    int[][] directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    int steps = 0;

    while (!queue.isEmpty()) {
        for (int levelSize = queue.size(); levelSize > 0; levelSize--) {
            int[] state = queue.poll();
            int row = state[0], col = state[1], keys = state[2];
            if (keys == allKeys) return steps;
            for (int[] direction : directions) {
                int nextRow = row + direction[0], nextCol = col + direction[1];
                if (nextRow < 0 || nextRow >= rows || nextCol < 0 || nextCol >= cols) continue;
                char cell = grid[nextRow].charAt(nextCol);
                if (cell == '#') continue;
                int nextKeys = keys;
                if (cell >= 'a' && cell <= 'f') nextKeys |= 1 << (cell - 'a');
                if (cell >= 'A' && cell <= 'F' && (keys & (1 << (cell - 'A'))) == 0) continue;
                if (!visited[nextRow][nextCol][nextKeys]) {
                    visited[nextRow][nextCol][nextKeys] = true;
                    queue.offer(new int[]{nextRow, nextCol, nextKeys});
                }
            }
        }
        steps++;
    }
    return -1;
}
```

**The unlock:** the vertex is `(row, col, keys)`, not merely `(row, col)`. Complexity is measured on the expanded state graph.

---

## G7. Eulerian Path - Reconstruct Itinerary

**Use when:** every edge/ticket/pair must be used exactly once.

```java
public List<String> findItinerary(List<List<String>> tickets) {
    Map<String, PriorityQueue<String>> graph = new HashMap<>();
    for (List<String> ticket : tickets) {
        graph.computeIfAbsent(ticket.get(0), ignored -> new PriorityQueue<>())
             .offer(ticket.get(1));
    }
    LinkedList<String> route = new LinkedList<>();
    buildRoute("JFK", graph, route);
    return route;
}

private void buildRoute(
    String airport,
    Map<String, PriorityQueue<String>> graph,
    LinkedList<String> route
) {
    PriorityQueue<String> destinations = graph.get(airport);
    while (destinations != null && !destinations.isEmpty()) {
        buildRoute(destinations.poll(), graph, route);
    }
    route.addFirst(airport);
}
```

Hierholzer builds the route backward by adding a vertex only when it has no unused outgoing edges. Eulerian means every edge once; Hamiltonian means every vertex once.

---

## Advanced Graph Selection Card

| Clue | Reach for |
| ---- | --------- |
| First/nearest/fewest steps; equal edges | BFS |
| Many sources spread simultaneously | Multi-source BFS |
| Weights only 0/1 | 0-1 BFS |
| Non-negative unequal weights | Dijkstra |
| Negative edges or bounded edge count | Bellman-Ford |
| Dependencies / DAG path or count | Topological sort + DP |
| Merge groups / redundant edge | Union-Find |
| Minimum total connection cost | MST |
| Critical edge | Low-link DFS |
| Use every edge once | Hierholzer |
| Keys, stops, coupons, parity, resource | Expand the state |

Always state: the vertex, the edge, direction, weight rules, when a distance is final, and complexity in V/E or expanded states.

---

<a id="section-python-vs-java-for-interviews"></a>

# Python vs Java for Interviews

## Quick Answer

**Practice both, but prioritize the language you'll use in the interview.** If you're targeting companies that require Java, focus 70% on Java, 30% on Python patterns.

---

## Python Advantages

```python
# Two Sum in Python
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
```

| Advantage                | Example                                   |
| ------------------------ | ----------------------------------------- |
| Less boilerplate         | No type declarations, shorter syntax      |
| Built-in data structures | `list`, `dict`, `set` work out of the box |
| Slicing                  | `arr[1:5]`, `arr[::-1]` for reverse       |
| List comprehensions      | `[x*2 for x in arr if x > 0]`             |
| Multiple return          | `return a, b`                             |
| Negative indexing        | `arr[-1]` for last element                |
| Easier heap              | `heapq` just works on lists               |

---

## Java Advantages

```java
// Two Sum in Java
public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> seen = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {
        int complement = target - nums[i];
        if (seen.containsKey(complement)) {
            return new int[] { seen.get(complement), i };
        }
        seen.put(nums[i], i);
    }
    return new int[] {};
}
```

| Advantage         | Example                               |
| ----------------- | ------------------------------------- |
| Type safety       | Compiler catches errors               |
| Explicit          | Easier to debug, clearer intent       |
| Industry standard | Many companies use Java in production |
| Performance       | Generally faster execution            |
| OOP strength      | Good for system design discussion     |

---

## Code Length Comparison

| Problem             | Python Lines | Java Lines |
| ------------------- | ------------ | ---------- |
| Two Sum             | 6            | 12         |
| Reverse Linked List | 7            | 12         |
| BFS Level Order     | 12           | 22         |
| Merge K Lists       | 15           | 28         |

Python is roughly **40-50% less code** on average.

---

## When to Use Each

### Use Python When:

- Company allows language choice
- Speed of coding matters (timed interview)
- Heavy on data manipulation
- You're more fluent in Python
- Interviewing at Python-heavy companies (ML/AI, startups)

### Use Java When:

- Company requires Java
- Interviewing at Java shops (banks, enterprise)
- You're more comfortable with Java
- Want to demonstrate OOP knowledge
- The role specifically uses Java

---

## Key Java Syntax to Memorize

```java
// HashMap
Map<Integer, Integer> map = new HashMap<>();
map.put(key, value);
map.get(key);
map.containsKey(key);
map.getOrDefault(key, defaultValue);
for (Map.Entry<K, V> entry : map.entrySet()) { }

// HashSet
Set<Integer> set = new HashSet<>();
set.add(value);
set.contains(value);
set.remove(value);

// ArrayList
List<Integer> list = new ArrayList<>();
list.add(value);
list.get(index);
list.set(index, value);
list.remove(index);
list.size();

// Queue
Queue<Integer> queue = new LinkedList<>();
queue.offer(value);  // add
queue.poll();        // remove and return
queue.peek();        // look at front

// Stack
Stack<Integer> stack = new Stack<>();
Deque<Integer> stack = new ArrayDeque<>();  // preferred
stack.push(value);
stack.pop();
stack.peek();

// PriorityQueue (min heap by default)
PriorityQueue<Integer> minHeap = new PriorityQueue<>();
PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[0] - b[0]);  // custom comparator
pq.offer(value);
pq.poll();
pq.peek();

// Array operations
Arrays.sort(arr);
Arrays.fill(arr, value);
int[] copy = Arrays.copyOf(arr, arr.length);

// String operations
s.charAt(i);
s.substring(start, end);
s.toCharArray();
String.valueOf(num);
Integer.parseInt(s);

// StringBuilder
StringBuilder sb = new StringBuilder();
sb.append("text");
sb.toString();
```

---

## Recommendation for Your Situation

If you might face a **Java-only interview**:

1. **Week 1:** Solve problems in Python first to understand patterns
2. **Week 1:** Immediately re-solve in Java to build muscle memory
3. **Week 2:** Solve directly in Java, use Python only if stuck

### Practice Strategy:

```
Day 1-3: Solve in Python → Translate to Java (learn syntax)
Day 4-7: Solve directly in Java (build speed)
Day 8-14: Time yourself in Java (target: Medium in 25 min)
```

### Key Things to Practice in Java:

- [ ] Creating and iterating HashMap/HashSet
- [ ] PriorityQueue with custom comparator
- [ ] Building adjacency list for graphs
- [ ] String manipulation (charAt, substring, StringBuilder)
- [ ] Array initialization and copying

---

## Final Verdict

> **Python is easier, but if the job requires Java, you need Java.**

Most top companies (Google, Meta, Amazon) let you choose. But banks (Goldman, JPMorgan), enterprise (Oracle, IBM), and some teams specifically want Java.

**Ask your recruiter** what languages are accepted. If Java is required, commit to it fully. The extra verbosity is annoying but manageable with practice.

The concepts are the same—only the syntax differs. Master patterns in any language, then translate.
