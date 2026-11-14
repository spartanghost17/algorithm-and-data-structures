# Complete DSA Patterns: Multiple Problems Per Pattern (Python)

<a id="quick-navigation"></a>

## Quick navigation

- [0. Python Language & Data Structure Cheat Sheet](#section-0-python-language-and-data-structure-cheat-sheet)
- [1. ARRAYS & HASHING](#section-1-arrays-and-hashing)
- [2. TWO POINTERS](#section-2-two-pointers)
- [3. SLIDING WINDOW](#section-3-sliding-window)
- [4. BINARY SEARCH](#section-4-binary-search)
- [5. STACKS](#section-5-stacks)
- [6. QUEUES](#section-6-queues)
- [7. LINKED LISTS](#section-7-linked-lists)
- [8. TREES - DFS](#section-8-trees-dfs)
- [9. TREES - BFS](#section-9-trees-bfs)
- [10. GRAPHS - DFS](#section-10-graphs-dfs)
- [11. GRAPHS - BFS (SHORTEST PATH)](#section-11-graphs-bfs-shortest-path)
- [12. TOPOLOGICAL SORT](#section-12-topological-sort)
- [13. HEAPS / PRIORITY QUEUE](#section-13-heaps-priority-queue)
- [14. BACKTRACKING](#section-14-backtracking)
- [15. ADVANCED GRAPH PATTERNS](#section-15-advanced-graph-patterns)
- [Quick Reference: Pattern Selection](#section-quick-reference-pattern-selection)


---

<a id="section-0-python-language-and-data-structure-cheat-sheet"></a>

## 0. Python Language & Data Structure Cheat Sheet

### Imports and basic syntax

```python
from collections import Counter, defaultdict, deque
import heapq
from math import inf

def solve(nums):
    answer = 0
    return answer

# Multiple assignment / swap
left, right = right, left
quotient, remainder = divmod(17, 5)
```

### Loops

```python
for value in nums:                         # values
    pass

for index in range(len(nums)):             # indices: 0 ... n - 1
    value = nums[index]

for index, value in enumerate(nums):       # index + value
    pass

for index in range(len(nums) - 1, -1, -1): # backwards by index
    pass

for value in reversed(nums):               # backwards by value
    pass

for left_value, right_value in zip(a, b):  # parallel iteration
    pass

for key, value in mapping.items():         # dictionary entries
    pass

while left <= right:
    left += 1

# break exits one loop; continue skips to its next iteration.
```

### Lists / dynamic arrays

```python
nums = [3, 1, 2]
zeros = [0] * n
matrix = [[0] * cols for _ in range(rows)] # independent rows

nums.append(4)             # add at end: O(1) amortised
nums.extend([5, 6])        # add many at end
nums.insert(1, 99)         # add at index: O(n)

last = nums.pop()          # remove + return last: O(1)
value = nums.pop(1)        # remove + return index: O(n)
nums.remove(99)            # remove first matching value: O(n), raises if absent
del nums[1]                # delete index
nums.clear()

exists = 3 in nums         # O(n)
index = nums.index(3)      # raises ValueError if absent
count = nums.count(3)

nums.sort()                # mutate ascending
nums.sort(reverse=True)
ordered = sorted(nums)     # new list
pairs.sort(key=lambda pair: (pair[1], pair[0]))

copy = nums.copy()         # shallow copy
piece = nums[start:end]    # end excluded
reversed_copy = nums[::-1]
squares = [value * value for value in nums if value > 0]
```

**Trap:** never create a matrix with `[[0] * cols] * rows`; every row would reference the same list.

### Tuples and strings

```python
point = (row, col)              # immutable; hashable if contents are hashable
row, col = point
first, *middle, last = nums

text = "graph"
character = text[index]
substring = text[start:end]     # end excluded
contains = "rap" in text
position = text.find("x")       # -1 if absent
parts = text.split(",")
joined = "-".join(parts)
trimmed = text.strip()
replaced = text.replace("a", "b")
number = int("42")
digits = str(42)

characters = list(text)         # strings are immutable
characters[index] = "X"
text = "".join(characters)
```

### Dictionaries / hash maps

```python
mapping = {}
mapping[key] = value                           # add or replace
value = mapping[key]                           # KeyError if missing
value = mapping.get(key, default)
mapping.setdefault(key, []).append(value)
frequency[value] = frequency.get(value, 0) + 1

exists = key in mapping                        # checks keys
removed = mapping.pop(key, default)
del mapping[key]                               # KeyError if missing

for key in mapping:
    value = mapping[key]
for key, value in mapping.items():
    pass

groups = defaultdict(list)
groups[key].append(value)
frequency = Counter(nums)
most_common = frequency.most_common(k)
```

Average lookup/add/remove is O(1). Keys must be hashable: numbers, strings, or immutable tuples work; lists do not.

### Sets

```python
seen = set()
seen.add(value)
seen.update(values)
exists = value in seen
seen.remove(value)       # raises KeyError if absent
seen.discard(value)      # safe if absent
value = seen.pop()       # removes an arbitrary item
seen.clear()

union = first | second
intersection = first & second
difference = first - second
is_subset = first <= second
```

Average membership/add/remove is O(1). Use a set for fast existence checks and uniqueness.

### Stack, queue, and deque

```python
# Stack: use a list
stack = []
stack.append(value)      # push
top = stack[-1]          # peek
value = stack.pop()      # pop
is_empty = not stack

# Queue / deque
queue = deque()
queue.append(value)      # add right
value = queue.popleft()  # remove left
front = queue[0]
queue.appendleft(value)
value = queue.pop()      # remove right
is_empty = not queue
```

Never use `list.pop(0)` for a queue—it is O(n). A `deque` gives O(1) operations at both ends.

### Heap / priority queue

```python
min_heap = []
heapq.heappush(min_heap, value)
smallest = min_heap[0]
smallest = heapq.heappop(min_heap)
smallest = heapq.heappushpop(min_heap, value)
heapq.heapify(nums)                         # in place, O(n)

max_heap = []
heapq.heappush(max_heap, -value)            # negate numeric priorities
largest = -heapq.heappop(max_heap)

task_heap = []
heapq.heappush(task_heap, (priority, node))  # tuples compare left to right
priority, node = heapq.heappop(task_heap)
```

Heap push/pop is O(log n); peek is O(1). Python's heap is a min-heap.

### Graphs, grids, and common interview helpers

```python
# Unweighted graph
graph = [[] for _ in range(n)]
for first, second in edges:
    graph[first].append(second)
    graph[second].append(first)  # omit for directed graphs

# Weighted graph: graph[u] contains (neighbour, weight)
weighted = [[] for _ in range(n)]
for source, target, weight in edges:
    weighted[source].append((target, weight))

visited = [False] * n
directions4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Safe infinity and copies
best = inf
negative_best = -inf
shallow = nums.copy()
grid_copy = [row[:] for row in grid]
```

### Complexity reminders

| Structure | Add | Remove | Membership / lookup | Peek |
| --------- | --- | ------ | ------------------- | ---- |
| `list` end | O(1)* | O(1) | O(n) | O(1) |
| `list` front/middle | O(n) | O(n) | O(n) | O(1) |
| `dict` / `set` average | O(1) | O(1) | O(1) | — |
| `deque` ends | O(1) | O(1) | O(n) | O(1) |
| heap | O(log n) | O(log n) | O(n) | O(1) |

*Amortised. Prefer clear operations over clever one-liners during interviews.

---


<a id="section-1-arrays-and-hashing"></a>

# 1. ARRAYS & HASHING

---

## Pattern 1A: Complement Lookup (Two Sum Pattern)

### Problem: Two Sum

**Statement:** Given array `nums` and `target`, return indices of two numbers that add up to target.

```
Input: nums = [2, 7, 11, 15], target = 9
Output: [0, 1]
```

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

**Keywords:** "two numbers that add/sum to", "pair equals"
**Intuition:** Store what you've seen. Check if the complement exists.

---

## Pattern 1B: Frequency Counting

### Problem: Top K Frequent Elements

**Statement:** Given array `nums` and integer `k`, return the `k` most frequent elements.

```
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1, 2]
```

```python
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    count = Counter(nums)

    # Bucket sort: index = frequency, value = list of nums with that frequency
    buckets = [[] for _ in range(len(nums) + 1)]

    for num, freq in count.items():
        buckets[freq].append(num)

    # Collect from highest frequency buckets
    result = []
    for freq in range(len(buckets) - 1, 0, -1):
        for num in buckets[freq]:
            result.append(num)
            if len(result) == k:
                return result

    return result
```

**Keywords:** "most frequent", "top k", "count occurrences"
**Intuition:** Count frequencies with hashmap. Use bucket sort to avoid O(n log n) sorting.

---

## Pattern 1C: Grouping by Key Transformation

### Problem: Group Anagrams

**Statement:** Given array of strings, group anagrams together.

```
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

```python
from collections import defaultdict

def group_anagrams(strs: list[str]) -> list[list[str]]:
    groups = defaultdict(list)

    for s in strs:
        # Key: sorted characters (anagrams have same sorted form)
        key = tuple(sorted(s))
        groups[key].append(s)

    return list(groups.values())
```

**Alternative Key (faster):**

```python
def group_anagrams_v2(strs: list[str]) -> list[list[str]]:
    groups = defaultdict(list)

    for s in strs:
        # Key: character count tuple
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        groups[tuple(count)].append(s)

    return list(groups.values())
```

**Keywords:** "group by", "anagrams", "same characters"
**Intuition:** Find a canonical form (key) that's identical for all items in a group.

---

## Pattern 1D: Prefix Sum + HashMap

### Problem: Subarray Sum Equals K

**Statement:** Given array `nums` and integer `k`, return count of subarrays that sum to `k`.

```
Input: nums = [1,1,1], k = 2
Output: 2 (subarrays [1,1] starting at index 0 and 1)
```

```python
def subarray_sum(nums: list[int], k: int) -> int:
    count = 0
    prefix_sum = 0
    seen = {0: 1}  # prefix_sum -> how many times seen

    for num in nums:
        prefix_sum += num

        # If (prefix_sum - k) was seen before, those subarrays sum to k
        if prefix_sum - k in seen:
            count += seen[prefix_sum - k]

        seen[prefix_sum] = seen.get(prefix_sum, 0) + 1

    return count
```

**Keywords:** "subarray sum equals", "contiguous elements sum to"
**Intuition:** prefix[j] - prefix[i] = sum of subarray [i+1, j]. Store prefix sums; look for prefix_sum - k.

---

<a id="section-2-two-pointers"></a>

# 2. TWO POINTERS

---

## Pattern 2A: Opposite Ends (Sorted or Optimization)

### Problem: Container With Most Water

**Statement:** Given `height[i]`, find two lines that form container holding most water.

```
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
```

```python
def max_area(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    max_water = 0

    while left < right:
        width = right - left
        h = min(height[left], height[right])
        max_water = max(max_water, width * h)

        # Move shorter line (only chance to improve)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_water
```

**Keywords:** "two lines", "container", "maximum area"
**Intuition:** Start widest. Move the shorter pointer; moving taller can never help.

---

## Pattern 2B: Opposite Ends (Sorted Array - 3Sum)

### Problem: 3Sum

**Statement:** Find all unique triplets that sum to zero.

```
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
```

```python
def three_sum(nums: list[int]) -> list[list[int]]:
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i-1]:
            continue

        # Two pointer for remaining two elements
        left, right = i + 1, len(nums) - 1
        target = -nums[i]

        while left < right:
            current_sum = nums[left] + nums[right]

            if current_sum == target:
                result.append([nums[i], nums[left], nums[right]])
                left += 1
                right -= 1
                # Skip duplicates
                while left < right and nums[left] == nums[left-1]:
                    left += 1
                while left < right and nums[right] == nums[right+1]:
                    right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1

    return result
```

**Keywords:** "three numbers sum to", "triplets", "unique"
**Intuition:** Sort. Fix one number, use two pointers to find the other two. Skip duplicates.

---

## Pattern 2C: Same Direction (Slow/Fast - Remove Duplicates)

### Problem: Remove Duplicates from Sorted Array

**Statement:** Remove duplicates in-place from sorted array. Return new length.

```
Input: nums = [1,1,2]
Output: 2 (nums becomes [1,2,...])
```

```python
def remove_duplicates(nums: list[int]) -> int:
    if not nums:
        return 0

    slow = 0  # Position to write next unique element

    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]

    return slow + 1
```

**Keywords:** "remove duplicates", "in-place", "sorted array"
**Intuition:** Slow marks write position. Fast scans for new unique elements.

---

## Pattern 2D: Same Direction (Slow/Fast - Move Zeroes)

### Problem: Move Zeroes

**Statement:** Move all 0's to end of array while maintaining order of non-zero elements.

```
Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]
```

```python
def move_zeroes(nums: list[int]) -> None:
    slow = 0  # Position for next non-zero

    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1
```

**Keywords:** "move zeroes", "maintain order", "in-place"
**Intuition:** Swap non-zero elements to the front. Zeroes naturally end up at the back.

---

<a id="section-3-sliding-window"></a>

# 3. SLIDING WINDOW

---

## Pattern 3A: Fixed Size Window

### Problem: Maximum Sum Subarray of Size K

**Statement:** Find maximum sum of any contiguous subarray of size `k`.

```
Input: nums = [2,1,5,1,3,2], k = 3
Output: 9 (subarray [5,1,3])
```

```python
def max_sum_subarray(nums: list[int], k: int) -> int:
    if len(nums) < k:
        return 0

    # Calculate first window
    window_sum = sum(nums[:k])
    max_sum = window_sum

    # Slide the window
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]  # Add new, remove old
        max_sum = max(max_sum, window_sum)

    return max_sum
```

**Keywords:** "subarray of size k", "window of size k", "consecutive k elements"
**Intuition:** Maintain sum of k elements. Slide by adding one, removing one.

---

## Pattern 3B: Variable Window - Longest Valid

### Problem: Longest Substring Without Repeating Characters

**Statement:** Find length of longest substring without repeating characters.

```
Input: s = "abcabcbb"
Output: 3 ("abc")
```

```python
def length_of_longest_substring(s: str) -> int:
    window = set()
    left = 0
    max_length = 0

    for right in range(len(s)):
        # Shrink until valid
        while s[right] in window:
            window.remove(s[left])
            left += 1

        window.add(s[right])
        max_length = max(max_length, right - left + 1)

    return max_length
```

**Keywords:** "longest substring", "without repeating", "at most k distinct"
**Intuition:** Expand right. When invalid, shrink left until valid. Track max.

---

## Pattern 3C: Variable Window - Longest with Constraint

### Problem: Longest Repeating Character Replacement

**Statement:** Given string `s` and integer `k`, you can replace at most `k` characters. Find longest substring with all same characters.

```
Input: s = "AABABBA", k = 1
Output: 4 ("AABA" -> "AAAA" with 1 replacement)
```

```python
def character_replacement(s: str, k: int) -> int:
    count = {}
    left = 0
    max_freq = 0  # Frequency of most common char in window
    max_length = 0

    for right in range(len(s)):
        count[s[right]] = count.get(s[right], 0) + 1
        max_freq = max(max_freq, count[s[right]])

        # Window size - max_freq = chars to replace
        # If > k, shrink window
        while (right - left + 1) - max_freq > k:
            count[s[left]] -= 1
            left += 1

        max_length = max(max_length, right - left + 1)

    return max_length
```

**Keywords:** "at most k replacements", "longest with k changes"
**Intuition:** Valid window: (window_size - max_freq) ≤ k. Expand, shrink when invalid.

---

## Pattern 3D: Variable Window - Shortest Valid

### Problem: Minimum Window Substring

**Statement:** Find shortest substring of `s` containing all characters of `t`.

```
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
```

```python
from collections import Counter

def min_window(s: str, t: str) -> str:
    if not t or not s:
        return ""

    need = Counter(t)
    have = {}
    required = len(need)  # Unique chars needed
    formed = 0  # Unique chars with enough count

    left = 0
    min_len = float('inf')
    result = (0, 0)

    for right in range(len(s)):
        char = s[right]
        have[char] = have.get(char, 0) + 1

        if char in need and have[char] == need[char]:
            formed += 1

        # Shrink while valid (to find minimum)
        while formed == required:
            # Update result
            if right - left + 1 < min_len:
                min_len = right - left + 1
                result = (left, right)

            # Shrink from left
            left_char = s[left]
            have[left_char] -= 1
            if left_char in need and have[left_char] < need[left_char]:
                formed -= 1
            left += 1

    return "" if min_len == float('inf') else s[result[0]:result[1]+1]
```

**Keywords:** "minimum window", "shortest substring containing", "smallest subarray with"
**Intuition:** Expand until valid. Then shrink while still valid, tracking minimum.

---

<a id="section-4-binary-search"></a>

# 4. BINARY SEARCH

---

## Pattern 4A: Standard Binary Search

### Problem: Binary Search

**Statement:** Find target in sorted array. Return index or -1.

```
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
```

```python
def binary_search(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

**Keywords:** "sorted array", "find element", "O(log n)"
**Intuition:** Compare mid, eliminate half each iteration.

---

## Pattern 4B: Find First/Last Occurrence (Boundaries)

### Problem: Find First and Last Position of Element

**Statement:** Find starting and ending position of target in sorted array.

```
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3, 4]
```

```python
def search_range(nums: list[int], target: int) -> list[int]:
    def find_left():
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left

    def find_right():
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        return right

    left_idx = find_left()
    right_idx = find_right()

    if left_idx <= right_idx and left_idx < len(nums) and nums[left_idx] == target:
        return [left_idx, right_idx]
    return [-1, -1]
```

**Keywords:** "first occurrence", "last occurrence", "leftmost/rightmost"
**Intuition:** Modify binary search to keep going even after finding target.

---

## Pattern 4C: Search in Rotated Sorted Array

### Problem: Search in Rotated Sorted Array

**Statement:** Array was sorted then rotated. Find target index.

```
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
```

```python
def search_rotated(nums: list[int], target: int) -> int:
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid

        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
```

**Keywords:** "rotated sorted", "pivot"
**Intuition:** One half is always sorted. Determine which, check if target is there.

---

## Pattern 4D: Binary Search on Answer

### Problem: Koko Eating Bananas

**Statement:** Koko has piles of bananas and h hours. Find minimum eating speed to finish.

```
Input: piles = [3,6,7,11], h = 8
Output: 4
```

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
            right = mid  # Try slower
        else:
            left = mid + 1  # Need faster

    return left
```

**Keywords:** "minimum speed/capacity that satisfies", "minimum X such that"
**Intuition:** Binary search on the answer space, not the input. Check if answer works.

---

<a id="section-5-stacks"></a>

# 5. STACKS

---

## Pattern 5A: Matching Pairs (Parentheses)

### Problem: Valid Parentheses

**Statement:** Determine if string of brackets is valid.

```
Input: s = "()[]{}"
Output: True

Input: s = "([)]"
Output: False
```

```python
def is_valid(s: str) -> bool:
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}

    for char in s:
        if char in pairs:  # Closing bracket
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
        else:  # Opening bracket
            stack.append(char)

    return len(stack) == 0
```

**Keywords:** "valid parentheses", "balanced brackets", "matching"
**Intuition:** Push opening brackets. Pop and match on closing brackets.

---

## Pattern 5B: Monotonic Stack - Next Greater Element

### Problem: Daily Temperatures

**Statement:** For each day, find days until warmer temperature.

```
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
```

```python
def daily_temperatures(temperatures: list[int]) -> list[int]:
    n = len(temperatures)
    answer = [0] * n
    stack = []  # Indices of days waiting for warmer

    for i in range(n):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            prev_day = stack.pop()
            answer[prev_day] = i - prev_day
        stack.append(i)

    return answer
```

**Keywords:** "next greater", "next warmer", "days until"
**Intuition:** Stack holds unresolved elements. Pop when current element resolves them.

---

## Pattern 5C: Monotonic Stack - Largest Rectangle

### Problem: Largest Rectangle in Histogram

**Statement:** Find largest rectangle area in histogram.

```
Input: heights = [2,1,5,6,2,3]
Output: 10 (rectangle of height 5, width 2)
```

```python
def largest_rectangle_area(heights: list[int]) -> int:
    stack = []  # (index, height)
    max_area = 0

    for i, h in enumerate(heights):
        start = i
        while stack and stack[-1][1] > h:
            idx, height = stack.pop()
            max_area = max(max_area, height * (i - idx))
            start = idx
        stack.append((start, h))

    # Process remaining in stack (extend to end)
    for idx, height in stack:
        max_area = max(max_area, height * (len(heights) - idx))

    return max_area
```

**Keywords:** "largest rectangle", "histogram", "maximum area"
**Intuition:** For each bar, find how far left and right it can extend. Monotonic stack finds boundaries.

---

## Pattern 5D: Min Stack (Design with O(1) Operations)

### Problem: Min Stack

**Statement:** Design stack supporting push, pop, top, and getMin in O(1).

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        # Push to min_stack if empty or val <= current min
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
```

**Keywords:** "min in O(1)", "design stack", "constant time minimum"
**Intuition:** Maintain parallel stack tracking minimum at each state.

---

<a id="section-6-queues"></a>

# 6. QUEUES

---

## Pattern 6A: BFS with Queue

### Problem: Rotting Oranges

**Statement:** Grid with 0=empty, 1=fresh, 2=rotten. Rot spreads each minute. Time until all rotten?

```
Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4
```

```python
from collections import deque

def oranges_rotting(grid: list[list[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0

    # Initialize
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c))
            elif grid[r][c] == 1:
                fresh += 1

    if fresh == 0:
        return 0

    minutes = 0
    directions = [(-1,0), (0,1), (1,0), (0,-1)]

    while queue:
        rotted = False
        for _ in range(len(queue)):
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    queue.append((nr, nc))
                    fresh -= 1
                    rotted = True
        if rotted:
            minutes += 1

    return minutes if fresh == 0 else -1
```

**Keywords:** "spreads", "minimum time", "level by level", "simultaneous"
**Intuition:** Multi-source BFS. All sources start together. Each level = 1 time unit.

---

## Pattern 6B: Monotonic Deque - Sliding Window Maximum

### Problem: Sliding Window Maximum

**Statement:** Return max of each sliding window of size k.

```
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
```

```python
from collections import deque

def max_sliding_window(nums: list[int], k: int) -> list[int]:
    result = []
    dq = deque()  # Indices, values in decreasing order

    for i in range(len(nums)):
        # Remove elements smaller than current (they'll never be max)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # Remove elements outside window
        if dq[0] <= i - k:
            dq.popleft()

        # Window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

**Keywords:** "sliding window maximum/minimum", "max of each window"
**Intuition:** Monotonic deque keeps potential maxes in decreasing order. Front is always max.

---

<a id="section-7-linked-lists"></a>

# 7. LINKED LISTS

---

## Pattern 7A: Reverse Linked List

### Problem: Reverse Linked List

**Statement:** Reverse a singly linked list.

```python
def reverse_list(head):
    prev = None
    curr = head

    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node

    return prev
```

**Keywords:** "reverse", "flip"
**Intuition:** Three pointers. Save next, reverse link, advance.

---

## Pattern 7B: Fast/Slow - Cycle Detection

### Problem: Linked List Cycle

**Statement:** Detect if linked list has a cycle.

```python
def has_cycle(head) -> bool:
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True

    return False
```

**Keywords:** "cycle", "loop", "circular"
**Intuition:** Fast catches up to slow if cycle exists.

---

## Pattern 7C: Fast/Slow - Find Middle

### Problem: Middle of Linked List

**Statement:** Return middle node.

```python
def middle_node(head):
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow
```

**Keywords:** "middle", "halfway"
**Intuition:** When fast reaches end, slow is at middle.

---

## Pattern 7D: Two Pointers with Gap

### Problem: Remove Nth Node From End

**Statement:** Remove nth node from end of list.

```
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]
```

```python
def remove_nth_from_end(head, n):
    dummy = ListNode(0, head)
    first = second = dummy

    # Advance first by n+1 steps
    for _ in range(n + 1):
        first = first.next

    # Move both until first reaches end
    while first:
        first = first.next
        second = second.next

    # Remove nth node
    second.next = second.next.next

    return dummy.next
```

**Keywords:** "nth from end", "kth from end"
**Intuition:** Create gap of n. When first hits end, second is at target.

---

## Pattern 7E: Merge Two Sorted Lists

### Problem: Merge Two Sorted Lists

**Statement:** Merge two sorted linked lists.

```python
def merge_two_lists(list1, list2):
    dummy = ListNode(0)
    curr = dummy

    while list1 and list2:
        if list1.val <= list2.val:
            curr.next = list1
            list1 = list1.next
        else:
            curr.next = list2
            list2 = list2.next
        curr = curr.next

    curr.next = list1 or list2

    return dummy.next
```

**Keywords:** "merge sorted", "combine"
**Intuition:** Compare heads, append smaller. Dummy node simplifies edge cases.

---

<a id="section-8-trees-dfs"></a>

# 8. TREES - DFS

---

## Pattern 8A: Basic DFS (Max Depth)

### Problem: Maximum Depth of Binary Tree

**Statement:** Find maximum depth of binary tree.

```python
def max_depth(root) -> int:
    if not root:
        return 0

    return 1 + max(max_depth(root.left), max_depth(root.right))
```

**Keywords:** "depth", "height"
**Intuition:** Depth = 1 + max depth of children.

---

## Pattern 8B: DFS with Validation (BST)

### Problem: Validate Binary Search Tree

**Statement:** Check if tree is valid BST.

```python
def is_valid_bst(root) -> bool:
    def validate(node, min_val, max_val):
        if not node:
            return True

        if node.val <= min_val or node.val >= max_val:
            return False

        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))

    return validate(root, float('-inf'), float('inf'))
```

**Alternative: Inorder traversal (should be ascending)**

```python
def is_valid_bst_inorder(root) -> bool:
    prev = float('-inf')

    def inorder(node):
        nonlocal prev
        if not node:
            return True

        if not inorder(node.left):
            return False

        if node.val <= prev:
            return False
        prev = node.val

        return inorder(node.right)

    return inorder(root)
```

**Keywords:** "valid BST", "binary search tree property"
**Intuition:** Pass valid range down. Or check inorder is ascending.

---

## Pattern 8C: DFS - Lowest Common Ancestor

### Problem: Lowest Common Ancestor

**Statement:** Find LCA of two nodes.

```python
def lowest_common_ancestor(root, p, q):
    if not root or root == p or root == q:
        return root

    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    if left and right:
        return root

    return left or right
```

**Keywords:** "common ancestor", "LCA"
**Intuition:** If both sides return non-null, current node is LCA.

---

## Pattern 8D: DFS - Path Sum

### Problem: Path Sum II

**Statement:** Find all root-to-leaf paths that sum to target.

```python
def path_sum(root, target_sum: int) -> list[list[int]]:
    result = []

    def dfs(node, remaining, path):
        if not node:
            return

        path.append(node.val)

        # Check if leaf with correct sum
        if not node.left and not node.right and remaining == node.val:
            result.append(path[:])

        dfs(node.left, remaining - node.val, path)
        dfs(node.right, remaining - node.val, path)

        path.pop()  # Backtrack

    dfs(root, target_sum, [])
    return result
```

**Keywords:** "path sum", "root to leaf"
**Intuition:** Track path while traversing. Backtrack when returning.

---

<a id="section-9-trees-bfs"></a>

# 9. TREES - BFS

---

## Pattern 9A: Level Order Traversal

### Problem: Binary Tree Level Order Traversal

**Statement:** Return values level by level.

```python
from collections import deque

def level_order(root) -> list[list[int]]:
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)

    return result
```

---

## Pattern 9B: Right Side View

### Problem: Binary Tree Right Side View

**Statement:** Return rightmost node at each level.

```python
from collections import deque

def right_side_view(root) -> list[int]:
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node = queue.popleft()
            if i == level_size - 1:  # Last node in level
                result.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    return result
```

**Keywords:** "right side view", "leftmost/rightmost each level"
**Intuition:** BFS, take last (or first) node of each level.

---

<a id="section-10-graphs-dfs"></a>

# 10. GRAPHS - DFS

---

## Pattern 10A: Flood Fill / Connected Components

### Problem: Number of Islands

**Statement:** Count islands in grid of '1' (land) and '0' (water).

```python
def num_islands(grid: list[list[str]]) -> int:
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return
        grid[r][c] = '0'  # Mark visited
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)

    return count
```

**Keywords:** "number of islands", "connected components", "regions"
**Intuition:** DFS from each unvisited land. Each DFS start = new island.

---

## Pattern 10B: Clone Graph

### Problem: Clone Graph

**Statement:** Deep copy a graph.

```python
def clone_graph(node):
    if not node:
        return None

    cloned = {}  # old node -> new node

    def dfs(node):
        if node in cloned:
            return cloned[node]

        copy = Node(node.val)
        cloned[node] = copy

        for neighbor in node.neighbors:
            copy.neighbors.append(dfs(neighbor))

        return copy

    return dfs(node)
```

**Keywords:** "clone", "deep copy", "duplicate graph"
**Intuition:** HashMap maps old nodes to clones. DFS to traverse and clone.

---

## Pattern 10C: DFS with Multiple Sources

### Problem: Pacific Atlantic Water Flow

**Statement:** Find cells that can reach both Pacific (top/left) and Atlantic (bottom/right) oceans.

```python
def pacific_atlantic(heights: list[list[int]]) -> list[list[int]]:
    if not heights:
        return []

    rows, cols = len(heights), len(heights[0])
    pacific = set()
    atlantic = set()

    def dfs(r, c, visited, prev_height):
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            (r, c) in visited or heights[r][c] < prev_height):
            return

        visited.add((r, c))
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            dfs(r + dr, c + dc, visited, heights[r][c])

    # DFS from Pacific edges (top and left)
    for c in range(cols):
        dfs(0, c, pacific, 0)
    for r in range(rows):
        dfs(r, 0, pacific, 0)

    # DFS from Atlantic edges (bottom and right)
    for c in range(cols):
        dfs(rows - 1, c, atlantic, 0)
    for r in range(rows):
        dfs(r, cols - 1, atlantic, 0)

    return list(pacific & atlantic)
```

**Keywords:** "reach both", "flow to", "multiple destinations"
**Intuition:** Reverse thinking. DFS from oceans inward. Find intersection.

---

<a id="section-11-graphs-bfs-shortest-path"></a>

# 11. GRAPHS - BFS (SHORTEST PATH)

---

## Pattern 11A: Shortest Path in Unweighted Graph

### Problem: Shortest Path in Binary Matrix

**Statement:** Shortest path from (0,0) to (n-1,n-1) in binary matrix. Can move 8 directions.

```python
from collections import deque

def shortest_path_binary_matrix(grid: list[list[int]]) -> int:
    n = len(grid)
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1

    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    queue = deque([(0, 0, 1)])
    grid[0][0] = 1

    while queue:
        r, c, dist = queue.popleft()

        if r == n-1 and c == n-1:
            return dist

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                grid[nr][nc] = 1
                queue.append((nr, nc, dist + 1))

    return -1
```

---

## Pattern 11B: Word Ladder

### Problem: Word Ladder

**Statement:** Transform beginWord to endWord, changing one letter at a time. Each intermediate word must be in wordList.

```
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5 (hit -> hot -> dot -> dog -> cog)
```

```python
from collections import deque

def ladder_length(begin_word: str, end_word: str, word_list: list[str]) -> int:
    word_set = set(word_list)
    if end_word not in word_set:
        return 0

    queue = deque([(begin_word, 1)])
    visited = {begin_word}

    while queue:
        word, length = queue.popleft()

        if word == end_word:
            return length

        # Try changing each character
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]
                if new_word in word_set and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, length + 1))

    return 0
```

**Keywords:** "transform", "one change at a time", "minimum transformations"
**Intuition:** BFS where neighbors are words differing by one letter.

---

<a id="section-12-topological-sort"></a>

# 12. TOPOLOGICAL SORT

---

## Pattern 12A: Course Schedule (Cycle Detection)

### Problem: Course Schedule

**Statement:** Can you finish all courses given prerequisites?

```python
from collections import deque, defaultdict

def can_finish(num_courses: int, prerequisites: list[list[int]]) -> bool:
    graph = defaultdict(list)
    indegree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        indegree[course] += 1

    queue = deque([i for i in range(num_courses) if indegree[i] == 0])
    completed = 0

    while queue:
        course = queue.popleft()
        completed += 1

        for next_course in graph[course]:
            indegree[next_course] -= 1
            if indegree[next_course] == 0:
                queue.append(next_course)

    return completed == num_courses
```

---

## Pattern 12B: Course Schedule II (Return Order)

### Problem: Course Schedule II

**Statement:** Return order to take courses. Empty if impossible.

```python
from collections import deque, defaultdict

def find_order(num_courses: int, prerequisites: list[list[int]]) -> list[int]:
    graph = defaultdict(list)
    indegree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        indegree[course] += 1

    queue = deque([i for i in range(num_courses) if indegree[i] == 0])
    order = []

    while queue:
        course = queue.popleft()
        order.append(course)

        for next_course in graph[course]:
            indegree[next_course] -= 1
            if indegree[next_course] == 0:
                queue.append(next_course)

    return order if len(order) == num_courses else []
```

**Keywords:** "prerequisites", "dependencies", "order of tasks"
**Intuition:** Kahn's algorithm. Process nodes with no dependencies first.

---

<a id="section-13-heaps-priority-queue"></a>

# 13. HEAPS / PRIORITY QUEUE

---

## Pattern 13A: Kth Largest Element

### Problem: Kth Largest Element in Array

**Statement:** Find kth largest element.

```python
import heapq

def find_kth_largest(nums: list[int], k: int) -> int:
    # Min heap of size k
    heap = []

    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)

    return heap[0]
```

**Keywords:** "kth largest", "kth smallest"
**Intuition:** Min heap of size k. After processing, top is kth largest.

---

## Pattern 13B: Merge K Sorted Lists

### Problem: Merge K Sorted Lists

**Statement:** Merge k sorted linked lists.

```python
import heapq

def merge_k_lists(lists):
    heap = []

    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))

    dummy = ListNode(0)
    curr = dummy

    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next

        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

**Keywords:** "merge k sorted"
**Intuition:** Heap holds heads of all lists. Extract min, add its next.

---

## Pattern 13C: Find Median from Data Stream (Two Heaps)

### Problem: Find Median from Data Stream

**Statement:** Design structure to add numbers and find median.

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # Max heap (negate values)
        self.large = []  # Min heap

    def addNum(self, num: int) -> None:
        # Add to max heap first
        heapq.heappush(self.small, -num)

        # Balance: largest of small should be <= smallest of large
        if self.small and self.large and -self.small[0] > self.large[0]:
            heapq.heappush(self.large, -heapq.heappop(self.small))

        # Balance sizes (small can have at most 1 more)
        if len(self.small) > len(self.large) + 1:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
```

**Keywords:** "median", "running median", "data stream"
**Intuition:** Two heaps split data in half. Max heap for smaller half, min heap for larger half.

---

<a id="section-14-backtracking"></a>

# 14. BACKTRACKING

---

## Pattern 14A: Subsets

### Problem: Subsets

**Statement:** Return all possible subsets.

```
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

```python
def subsets(nums: list[int]) -> list[list[int]]:
    result = []

    def backtrack(start, current):
        result.append(current[:])  # Add copy of current subset

        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result
```

**Keywords:** "all subsets", "power set", "all combinations"
**Intuition:** At each position, choose to include or not. Start index prevents duplicates.

---

## Pattern 14B: Permutations

### Problem: Permutations

**Statement:** Return all possible permutations.

```
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

```python
def permutations(nums: list[int]) -> list[list[int]]:
    result = []

    def backtrack(current, remaining):
        if not remaining:
            result.append(current[:])
            return

        for i in range(len(remaining)):
            current.append(remaining[i])
            backtrack(current, remaining[:i] + remaining[i+1:])
            current.pop()

    backtrack([], nums)
    return result
```

**Alternative with visited set:**

```python
def permutations_v2(nums: list[int]) -> list[list[int]]:
    result = []
    used = [False] * len(nums)

    def backtrack(current):
        if len(current) == len(nums):
            result.append(current[:])
            return

        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            current.append(nums[i])
            backtrack(current)
            current.pop()
            used[i] = False

    backtrack([])
    return result
```

**Keywords:** "all permutations", "all arrangements", "order matters"
**Intuition:** Each position tries all remaining elements. Track used elements.

---

## Pattern 14C: Combination Sum

### Problem: Combination Sum

**Statement:** Find all combinations that sum to target. Numbers can be reused.

```python
def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    result = []

    def backtrack(start, remaining, current):
        if remaining == 0:
            result.append(current[:])
            return
        if remaining < 0:
            return

        for i in range(start, len(candidates)):
            current.append(candidates[i])
            backtrack(i, remaining - candidates[i], current)  # i, not i+1 (reuse allowed)
            current.pop()

    backtrack(0, target, [])
    return result
```

---

## Pattern 14D: Word Search

### Problem: Word Search

**Statement:** Check if word exists in grid (adjacent cells, no reuse).

```python
def exist(board: list[list[str]], word: str) -> bool:
    rows, cols = len(board), len(board[0])

    def backtrack(r, c, i):
        if i == len(word):
            return True

        if (r < 0 or r >= rows or c < 0 or c >= cols or
            board[r][c] != word[i]):
            return False

        # Mark visited
        temp = board[r][c]
        board[r][c] = '#'

        # Explore
        found = (backtrack(r+1, c, i+1) or
                 backtrack(r-1, c, i+1) or
                 backtrack(r, c+1, i+1) or
                 backtrack(r, c-1, i+1))

        # Restore
        board[r][c] = temp

        return found

    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    return False
```

**Keywords:** "word search", "path in grid", "spell word"
**Intuition:** DFS with backtracking. Mark visited, explore, restore.

---

## Pattern 14E: N-Queens

### Problem: N-Queens

**Statement:** Place n queens on n×n board so none attack each other.

```python
def solve_n_queens(n: int) -> list[list[str]]:
    result = []
    cols = set()
    pos_diag = set()  # r + c
    neg_diag = set()  # r - c

    board = [['.'] * n for _ in range(n)]

    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return

        for col in range(n):
            if col in cols or (row + col) in pos_diag or (row - col) in neg_diag:
                continue

            # Place queen
            cols.add(col)
            pos_diag.add(row + col)
            neg_diag.add(row - col)
            board[row][col] = 'Q'

            backtrack(row + 1)

            # Remove queen
            cols.remove(col)
            pos_diag.remove(row + col)
            neg_diag.remove(row - col)
            board[row][col] = '.'

    backtrack(0)
    return result
```

**Keywords:** "n-queens", "place without conflict", "constraint satisfaction"
**Intuition:** Place row by row. Track attacked columns and diagonals.

---

<a id="section-15-advanced-graph-patterns"></a>

# 15. ADVANCED GRAPH PATTERNS

Use this decision table before reaching for a template:

| Signal | Algorithm | Complexity |
| ------ | --------- | ---------- |
| Unweighted shortest path | BFS | O(V + E) |
| Many sources start together | Multi-source BFS | O(V + E) |
| Weights are only 0 or 1 | 0-1 BFS | O(V + E) |
| Non-negative weights | Dijkstra | O((V + E) log V) |
| Negative edges or at most K edges | Bellman-Ford | O(VE), or O(KE) |
| All-pairs distances, small graph | Floyd-Warshall | O(V^3) |
| Repeated merges/connectivity | Union-Find | Almost O(1) per operation |
| Minimum total cost to connect all nodes | Kruskal / Prim MST | O(E log E) / O(E log V) |
| Critical edges / strongly connected regions | Low-link DFS | O(V + E) |
| Use every edge exactly once | Hierholzer | O(E) plus edge ordering |
| Position plus keys/stops/resources | BFS/Dijkstra on full state | Depends on state count |

---

## Pattern 15A: Dijkstra - Non-Negative Weighted Shortest Path

### Problem: Network Delay Time

```python
from collections import defaultdict
import heapq

def networkDelayTime(times, n, k):
    graph = defaultdict(list)
    for u, v, weight in times:
        graph[u].append((v, weight))

    distances = [float("inf")] * (n + 1)
    distances[k] = 0
    min_heap = [(0, k)]  # (distance, node)

    while min_heap:
        distance, node = heapq.heappop(min_heap)

        # A better route was pushed after this stale entry.
        if distance != distances[node]:
            continue

        for neighbour, weight in graph[node]:
            candidate = distance + weight
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                heapq.heappush(min_heap, (candidate, neighbour))

    answer = max(distances[1:])
    return -1 if answer == float("inf") else answer
```

**Keywords:** "weighted shortest path", "minimum travel time", "non-negative costs"

**Invariant:** when the current best distance is removed from the min-heap, no later path can improve it through non-negative edges. Do not mark a node visited when it is first **inserted**.

**Complexity:** O((V + E) log V) time, O(V + E) space.

---

## Pattern 15B: 0-1 BFS - Binary Edge Weights

### Problem: Minimum Obstacle Removal to Reach Corner

```python
from collections import deque

def minimumObstacles(grid):
    rows, cols = len(grid), len(grid[0])
    distances = [[float("inf")] * cols for _ in range(rows)]
    distances[0][0] = 0
    queue = deque([(0, 0)])

    for_pop = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while queue:
        row, col = queue.popleft()
        for dr, dc in for_pop:
            next_row, next_col = row + dr, col + dc
            if not (0 <= next_row < rows and 0 <= next_col < cols):
                continue

            weight = grid[next_row][next_col]
            candidate = distances[row][col] + weight
            if candidate >= distances[next_row][next_col]:
                continue

            distances[next_row][next_col] = candidate
            if weight == 0:
                queue.appendleft((next_row, next_col))
            else:
                queue.append((next_row, next_col))

    return distances[rows - 1][cols - 1]
```

**Keywords:** "0 or 1 cost", "free versus paid edge", "minimum reversals/obstacles"

**Intuition:** a zero-cost transition belongs at the front; a cost-one transition belongs at the back. The deque maintains Dijkstra's distance order without a heap.

**Complexity:** O(V + E) time and O(V) space.

---

## Pattern 15C: Bellman-Ford by Layers - Bounded Stops

### Problem: Cheapest Flights Within K Stops

```python
def findCheapestPrice(n, flights, src, dst, k):
    costs = [float("inf")] * n
    costs[src] = 0

    # At most k stops means at most k + 1 edges.
    for _ in range(k + 1):
        next_costs = costs.copy()
        for source, target, price in flights:
            if costs[source] != float("inf"):
                next_costs[target] = min(
                    next_costs[target], costs[source] + price
                )
        costs = next_costs

    return -1 if costs[dst] == float("inf") else costs[dst]
```

**Keywords:** "negative weights", "at most K edges/stops", "relax every edge"

**Critical detail:** read from `costs` and write to `next_costs`. Updating one array in place could chain multiple flights in the same round and violate the edge limit.

**Complexity:** O(KE) time and O(V) space.

---

## Pattern 15D: Union-Find - Dynamic Connectivity and Cycle Detection

### Problem: Redundant Connection

```python
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.component_size = [1] * size

    def find(self, node):
        while node != self.parent[node]:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, first, second):
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return False

        if self.component_size[root_first] < self.component_size[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        self.component_size[root_first] += self.component_size[root_second]
        return True

def findRedundantConnection(edges):
    union_find = UnionFind(len(edges) + 1)
    for first, second in edges:
        if not union_find.union(first, second):
            return [first, second]
    return []
```

**Keywords:** "merge groups", "redundant edge", "connected after additions", "same component"

**Intuition:** `union` returns false exactly when an undirected edge closes a cycle. Path compression plus union by size gives near-constant amortised operations.

---

## Pattern 15E: Prim's Minimum Spanning Tree

### Problem: Min Cost to Connect All Points

```python
import heapq

def minCostConnectPoints(points):
    number_of_points = len(points)
    in_tree = [False] * number_of_points
    min_heap = [(0, 0)]  # (edge cost, destination)
    total_cost = 0
    used = 0

    while used < number_of_points:
        cost, point = heapq.heappop(min_heap)
        if in_tree[point]:
            continue

        in_tree[point] = True
        total_cost += cost
        used += 1

        x1, y1 = points[point]
        for neighbour, (x2, y2) in enumerate(points):
            if not in_tree[neighbour]:
                distance = abs(x1 - x2) + abs(y1 - y2)
                heapq.heappush(min_heap, (distance, neighbour))

    return total_cost
```

**Keywords:** "connect every node", "minimum total cost", "network/cable plan"

**MST versus shortest path:** an MST minimises the sum of chosen edges across the whole network; Dijkstra minimises routes from one source. They answer different questions.

**Complexity:** O(V^2 log V) for this implicit complete graph; standard adjacency-list Prim is O(E log V).

---

## Pattern 15F: Bipartite Graph - Two-Colouring

### Problem: Is Graph Bipartite?

```python
from collections import deque

def isBipartite(graph):
    colour = [-1] * len(graph)

    for start in range(len(graph)):  # handles disconnected graphs
        if colour[start] != -1:
            continue

        colour[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbour in graph[node]:
                if colour[neighbour] == -1:
                    colour[neighbour] = colour[node] ^ 1
                    queue.append(neighbour)
                elif colour[neighbour] == colour[node]:
                    return False

    return True
```

**Keywords:** "split into two groups", "enemies", "alternate", "odd cycle"

**Invariant:** every edge must cross from colour 0 to colour 1. An undirected graph is bipartite exactly when it has no odd-length cycle.

---

## Pattern 15G: Tarjan Low-Link - Bridges

### Problem: Critical Connections in a Network

```python
def criticalConnections(n, connections):
    graph = [[] for _ in range(n)]
    for edge_id, (first, second) in enumerate(connections):
        graph[first].append((second, edge_id))
        graph[second].append((first, edge_id))

    discovery = [-1] * n
    low = [0] * n
    timer = 0
    bridges = []

    def dfs(node, parent_edge):
        nonlocal timer
        discovery[node] = low[node] = timer
        timer += 1

        for neighbour, edge_id in graph[node]:
            if edge_id == parent_edge:
                continue
            if discovery[neighbour] == -1:
                dfs(neighbour, edge_id)
                low[node] = min(low[node], low[neighbour])
                if low[neighbour] > discovery[node]:
                    bridges.append([node, neighbour])
            else:
                low[node] = min(low[node], discovery[neighbour])

    for node in range(n):
        if discovery[node] == -1:
            dfs(node, -1)
    return bridges
```

**Keywords:** "critical connection", "single point of failure", "disconnect if removed"

**Meaning of `low[v]`:** the earliest discovery time reachable from `v`'s DFS subtree without using its parent edge. If `low[v] > discovery[u]`, the edge `u-v` has no alternate route and is a bridge.

**Complexity:** O(V + E) time and O(V + E) space.

---

## Pattern 15H: State-Space BFS - Node Plus Bitmask

### Problem: Shortest Path to Get All Keys

```python
from collections import deque

def shortestPathAllKeys(grid):
    rows, cols = len(grid), len(grid[0])
    all_keys = 0
    start = None

    for row in range(rows):
        for col in range(cols):
            cell = grid[row][col]
            if cell == "@":
                start = (row, col)
            elif "a" <= cell <= "f":
                all_keys |= 1 << (ord(cell) - ord("a"))

    queue = deque([(start[0], start[1], 0, 0)])  # row, col, mask, steps
    visited = {(start[0], start[1], 0)}

    while queue:
        row, col, keys, steps = queue.popleft()
        if keys == all_keys:
            return steps

        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row, next_col = row + dr, col + dc
            if not (0 <= next_row < rows and 0 <= next_col < cols):
                continue
            cell = grid[next_row][next_col]
            if cell == "#":
                continue

            next_keys = keys
            if "a" <= cell <= "f":
                next_keys |= 1 << (ord(cell) - ord("a"))
            if "A" <= cell <= "F" and not (
                keys & (1 << (ord(cell) - ord("A")))
            ):
                continue

            state = (next_row, next_col, next_keys)
            if state not in visited:
                visited.add(state)
                queue.append((next_row, next_col, next_keys, steps + 1))

    return -1
```

**Keywords:** "keys/locks", "collect all", "remaining eliminations", "coupons/stops"

**The hard-problem unlock:** `(row, col)` is not the vertex; `(row, col, keys)` is. Reaching the same cell with different keys creates different future moves.

---

## Pattern 15I: Eulerian Path - Use Every Edge Once

### Problem: Reconstruct Itinerary

```python
from collections import defaultdict

def findItinerary(tickets):
    graph = defaultdict(list)
    # Reverse sorting lets pop() take the smallest lexical destination.
    for source, destination in sorted(tickets, reverse=True):
        graph[source].append(destination)

    route = []

    def visit(airport):
        while graph[airport]:
            visit(graph[airport].pop())
        route.append(airport)

    visit("JFK")
    return route[::-1]
```

**Keywords:** "use every ticket/edge exactly once", "arrange pairs", "valid itinerary"

**Intuition:** Hierholzer consumes edges until stuck, then appends nodes during backtracking. The route is therefore built backwards. Eulerian uses every **edge** once; Hamiltonian uses every **vertex** once.

---

## Advanced Graph Interview Checklist

1. Define the real vertex and edge—including hidden state.
2. State whether edges are directed, weighted, and possibly negative.
3. Match weights to BFS, 0-1 BFS, Dijkstra, Bellman-Ford, or DAG relaxation.
4. Build both directions only for undirected edges.
5. Cover every component when no single source is guaranteed.
6. In Dijkstra, skip stale heap entries; in BFS, mark visited when enqueuing.
7. Distinguish MST total cost from single-source shortest paths.
8. Use iterative DFS if recursion depth can exceed Python's limit.
9. Give complexity in V and E—or the expanded state count.

**Next drills:** Accounts Merge (DSU), Swim in Rising Water (Dijkstra), Alien Dictionary (topological sort), Network Delay Time, Min Cost to Connect All Points, Critical Connections, Reconstruct Itinerary, and Shortest Path to Get All Keys.

---

<a id="section-quick-reference-pattern-selection"></a>

# Quick Reference: Pattern Selection

| Keywords                        | Pattern                              |
| ------------------------------- | ------------------------------------ |
| "pair/two sum to"               | HashMap complement                   |
| "frequency/count/most common"   | HashMap counting                     |
| "group by/anagrams"             | HashMap with key transform           |
| "subarray sum equals"           | Prefix sum + HashMap                 |
| "two pointers + sorted"         | Opposite ends                        |
| "triplet/3sum"                  | Sort + two pointers                  |
| "remove duplicates in-place"    | Slow/fast pointers                   |
| "substring" + "longest"         | Variable sliding window              |
| "subarray of size k"            | Fixed sliding window                 |
| "minimum window containing"     | Variable window (shrink while valid) |
| "sorted + find"                 | Binary search                        |
| "minimum X that satisfies"      | Binary search on answer              |
| "valid parentheses"             | Stack matching                       |
| "next greater/smaller"          | Monotonic stack                      |
| "level by level/BFS"            | Queue                                |
| "sliding window max"            | Monotonic deque                      |
| "reverse/cycle in list"         | Linked list pointers                 |
| "tree depth/validate"           | DFS                                  |
| "level order"                   | BFS                                  |
| "islands/connected"             | DFS flood fill                       |
| "shortest path unweighted"      | BFS                                  |
| "prerequisites/dependencies"    | Topological sort                     |
| "weighted shortest path"        | Dijkstra                             |
| "weights are only 0/1"          | 0-1 BFS                              |
| "negative edges/at most k edges"| Bellman-Ford                         |
| "merge groups/redundant edge"   | Union-Find                           |
| "minimum total connection cost" | Minimum spanning tree                |
| "critical connection"           | Tarjan low-link                      |
| "use every edge exactly once"   | Eulerian path / Hierholzer           |
| "node plus keys/stops/resources"| State-space graph                    |
| "kth largest/merge k sorted"    | Heap                                 |
| "all combinations/permutations" | Backtracking                         |
