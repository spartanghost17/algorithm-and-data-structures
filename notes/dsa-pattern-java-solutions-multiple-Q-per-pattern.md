# Complete DSA Patterns: Multiple Problems Per Pattern (Java)

<a id="quick-navigation"></a>

## Quick navigation

- [0. Java Language & Data Structure Cheat Sheet](#section-0-java-language-and-data-structure-cheat-sheet)
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
- [Java Syntax Cheat Sheet](#section-java-syntax-cheat-sheet)


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

# 1. ARRAYS & HASHING

---

## Pattern 1A: Complement Lookup (Two Sum Pattern)

### Problem: Two Sum

**Statement:** Given array `nums` and `target`, return indices of two numbers that add up to target.

```
Input: nums = [2, 7, 11, 15], target = 9
Output: [0, 1]
```

```java
public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> seen = new HashMap<>();  // value -> index

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

```java
public int[] topKFrequent(int[] nums, int k) {
    // Count frequencies
    Map<Integer, Integer> count = new HashMap<>();
    for (int num : nums) {
        count.put(num, count.getOrDefault(num, 0) + 1);
    }

    // Bucket sort: index = frequency
    List<Integer>[] buckets = new List[nums.length + 1];
    for (int i = 0; i < buckets.length; i++) {
        buckets[i] = new ArrayList<>();
    }

    for (Map.Entry<Integer, Integer> entry : count.entrySet()) {
        buckets[entry.getValue()].add(entry.getKey());
    }

    // Collect from highest frequency
    int[] result = new int[k];
    int idx = 0;
    for (int freq = buckets.length - 1; freq >= 0 && idx < k; freq--) {
        for (int num : buckets[freq]) {
            result[idx++] = num;
            if (idx == k) break;
        }
    }

    return result;
}
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

```java
public List<List<String>> groupAnagrams(String[] strs) {
    Map<String, List<String>> groups = new HashMap<>();

    for (String s : strs) {
        char[] chars = s.toCharArray();
        Arrays.sort(chars);
        String key = new String(chars);

        groups.computeIfAbsent(key, k -> new ArrayList<>()).add(s);
    }

    return new ArrayList<>(groups.values());
}
```

**Alternative Key (character count):**

```java
public List<List<String>> groupAnagramsV2(String[] strs) {
    Map<String, List<String>> groups = new HashMap<>();

    for (String s : strs) {
        int[] count = new int[26];
        for (char c : s.toCharArray()) {
            count[c - 'a']++;
        }
        String key = Arrays.toString(count);

        groups.computeIfAbsent(key, k -> new ArrayList<>()).add(s);
    }

    return new ArrayList<>(groups.values());
}
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

```java
public int subarraySum(int[] nums, int k) {
    int count = 0;
    int prefixSum = 0;
    Map<Integer, Integer> seen = new HashMap<>();
    seen.put(0, 1);  // prefix_sum -> how many times seen

    for (int num : nums) {
        prefixSum += num;

        // If (prefixSum - k) was seen before, those subarrays sum to k
        if (seen.containsKey(prefixSum - k)) {
            count += seen.get(prefixSum - k);
        }

        seen.put(prefixSum, seen.getOrDefault(prefixSum, 0) + 1);
    }

    return count;
}
```

**Keywords:** "subarray sum equals", "contiguous elements sum to"
**Intuition:** prefix[j] - prefix[i] = sum of subarray [i+1, j]. Store prefix sums; look for prefixSum - k.

---

<a id="section-2-two-pointers"></a>

# 2. TWO POINTERS

---

## Pattern 2A: Opposite Ends (Sorted or Optimization)

### Problem: Container With Most Water

**Statement:** Given `height[i]`, find two lines that form container holding most water.

```java
public int maxArea(int[] height) {
    int left = 0, right = height.length - 1;
    int maxWater = 0;

    while (left < right) {
        int width = right - left;
        int h = Math.min(height[left], height[right]);
        maxWater = Math.max(maxWater, width * h);

        if (height[left] < height[right]) {
            left++;
        } else {
            right--;
        }
    }

    return maxWater;
}
```

**Keywords:** "two lines", "container", "maximum area"
**Intuition:** Start widest. Move the shorter pointer; moving taller can never help.

---

## Pattern 2B: Opposite Ends (Sorted Array - 3Sum)

### Problem: 3Sum

**Statement:** Find all unique triplets that sum to zero.

```java
public List<List<Integer>> threeSum(int[] nums) {
    Arrays.sort(nums);
    List<List<Integer>> result = new ArrayList<>();

    for (int i = 0; i < nums.length - 2; i++) {
        // Skip duplicates
        if (i > 0 && nums[i] == nums[i - 1]) continue;

        int left = i + 1, right = nums.length - 1;
        int target = -nums[i];

        while (left < right) {
            int sum = nums[left] + nums[right];

            if (sum == target) {
                result.add(Arrays.asList(nums[i], nums[left], nums[right]));
                left++;
                right--;

                // Skip duplicates
                while (left < right && nums[left] == nums[left - 1]) left++;
                while (left < right && nums[right] == nums[right + 1]) right--;
            } else if (sum < target) {
                left++;
            } else {
                right--;
            }
        }
    }

    return result;
}
```

**Keywords:** "three numbers sum to", "triplets", "unique"
**Intuition:** Sort. Fix one number, use two pointers to find the other two. Skip duplicates.

---

## Pattern 2C: Same Direction (Slow/Fast - Remove Duplicates)

### Problem: Remove Duplicates from Sorted Array

**Statement:** Remove duplicates in-place from sorted array. Return new length.

```java
public int removeDuplicates(int[] nums) {
    if (nums.length == 0) return 0;

    int slow = 0;

    for (int fast = 1; fast < nums.length; fast++) {
        if (nums[fast] != nums[slow]) {
            slow++;
            nums[slow] = nums[fast];
        }
    }

    return slow + 1;
}
```

**Keywords:** "remove duplicates", "in-place", "sorted array"
**Intuition:** Slow marks write position. Fast scans for new unique elements.

---

## Pattern 2D: Same Direction (Slow/Fast - Move Zeroes)

### Problem: Move Zeroes

**Statement:** Move all 0's to end of array while maintaining order of non-zero elements.

```java
public void moveZeroes(int[] nums) {
    int slow = 0;

    for (int fast = 0; fast < nums.length; fast++) {
        if (nums[fast] != 0) {
            int temp = nums[slow];
            nums[slow] = nums[fast];
            nums[fast] = temp;
            slow++;
        }
    }
}
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

```java
public int maxSumSubarray(int[] nums, int k) {
    if (nums.length < k) return 0;

    // Calculate first window
    int windowSum = 0;
    for (int i = 0; i < k; i++) {
        windowSum += nums[i];
    }
    int maxSum = windowSum;

    // Slide the window
    for (int i = k; i < nums.length; i++) {
        windowSum += nums[i] - nums[i - k];
        maxSum = Math.max(maxSum, windowSum);
    }

    return maxSum;
}
```

**Keywords:** "subarray of size k", "window of size k", "consecutive k elements"
**Intuition:** Maintain sum of k elements. Slide by adding one, removing one.

---

## Pattern 3B: Variable Window - Longest Valid

### Problem: Longest Substring Without Repeating Characters

**Statement:** Find length of longest substring without repeating characters.

```java
public int lengthOfLongestSubstring(String s) {
    Set<Character> window = new HashSet<>();
    int left = 0;
    int maxLength = 0;

    for (int right = 0; right < s.length(); right++) {
        while (window.contains(s.charAt(right))) {
            window.remove(s.charAt(left));
            left++;
        }

        window.add(s.charAt(right));
        maxLength = Math.max(maxLength, right - left + 1);
    }

    return maxLength;
}
```

**Keywords:** "longest substring", "without repeating", "at most k distinct"
**Intuition:** Expand right. When invalid, shrink left until valid. Track max.

---

## Pattern 3C: Variable Window - Longest with Constraint

### Problem: Longest Repeating Character Replacement

**Statement:** Given string `s` and integer `k`, you can replace at most `k` characters. Find longest substring with all same characters.

```java
public int characterReplacement(String s, int k) {
    int[] count = new int[26];
    int left = 0;
    int maxFreq = 0;
    int maxLength = 0;

    for (int right = 0; right < s.length(); right++) {
        count[s.charAt(right) - 'A']++;
        maxFreq = Math.max(maxFreq, count[s.charAt(right) - 'A']);

        // Window size - maxFreq = chars to replace
        while ((right - left + 1) - maxFreq > k) {
            count[s.charAt(left) - 'A']--;
            left++;
        }

        maxLength = Math.max(maxLength, right - left + 1);
    }

    return maxLength;
}
```

**Keywords:** "at most k replacements", "longest with k changes"
**Intuition:** Valid window: (window_size - max_freq) ≤ k. Expand, shrink when invalid.

---

## Pattern 3D: Variable Window - Shortest Valid

### Problem: Minimum Window Substring

**Statement:** Find shortest substring of `s` containing all characters of `t`.

```java
public String minWindow(String s, String t) {
    if (s.isEmpty() || t.isEmpty()) return "";

    Map<Character, Integer> need = new HashMap<>();
    for (char c : t.toCharArray()) {
        need.put(c, need.getOrDefault(c, 0) + 1);
    }

    Map<Character, Integer> have = new HashMap<>();
    int required = need.size();
    int formed = 0;

    int left = 0;
    int minLen = Integer.MAX_VALUE;
    int resultLeft = 0, resultRight = 0;

    for (int right = 0; right < s.length(); right++) {
        char c = s.charAt(right);
        have.put(c, have.getOrDefault(c, 0) + 1);

        if (need.containsKey(c) && have.get(c).equals(need.get(c))) {
            formed++;
        }

        while (formed == required) {
            if (right - left + 1 < minLen) {
                minLen = right - left + 1;
                resultLeft = left;
                resultRight = right;
            }

            char leftChar = s.charAt(left);
            have.put(leftChar, have.get(leftChar) - 1);
            if (need.containsKey(leftChar) && have.get(leftChar) < need.get(leftChar)) {
                formed--;
            }
            left++;
        }
    }

    return minLen == Integer.MAX_VALUE ? "" : s.substring(resultLeft, resultRight + 1);
}
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

```java
public int search(int[] nums, int target) {
    int left = 0, right = nums.length - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (nums[mid] == target) {
            return mid;
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}
```

**Keywords:** "sorted array", "find element", "O(log n)"
**Intuition:** Compare mid, eliminate half each iteration.

---

## Pattern 4B: Find First/Last Occurrence (Boundaries)

### Problem: Find First and Last Position of Element

**Statement:** Find starting and ending position of target in sorted array.

```java
public int[] searchRange(int[] nums, int target) {
    return new int[] { findLeft(nums, target), findRight(nums, target) };
}

private int findLeft(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    int result = -1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            result = mid;
            right = mid - 1;  // Keep searching left
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return result;
}

private int findRight(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    int result = -1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            result = mid;
            left = mid + 1;  // Keep searching right
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return result;
}
```

**Keywords:** "first occurrence", "last occurrence", "leftmost/rightmost"
**Intuition:** Modify binary search to keep going even after finding target.

---

## Pattern 4C: Search in Rotated Sorted Array

### Problem: Search in Rotated Sorted Array

**Statement:** Array was sorted then rotated. Find target index.

```java
public int searchRotated(int[] nums, int target) {
    int left = 0, right = nums.length - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (nums[mid] == target) {
            return mid;
        }

        // Left half is sorted
        if (nums[left] <= nums[mid]) {
            if (target >= nums[left] && target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        // Right half is sorted
        else {
            if (target > nums[mid] && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }

    return -1;
}
```

**Keywords:** "rotated sorted", "pivot"
**Intuition:** One half is always sorted. Determine which, check if target is there.

---

## Pattern 4D: Binary Search on Answer

### Problem: Koko Eating Bananas

**Statement:** Koko has piles of bananas and h hours. Find minimum eating speed to finish.

```java
public int minEatingSpeed(int[] piles, int h) {
    int left = 1;
    int right = Arrays.stream(piles).max().getAsInt();

    while (left < right) {
        int mid = left + (right - left) / 2;

        if (canFinish(piles, mid, h)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }

    return left;
}

private boolean canFinish(int[] piles, int speed, int h) {
    int hours = 0;
    for (int pile : piles) {
        hours += (pile + speed - 1) / speed;  // Ceiling division
    }
    return hours <= h;
}
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

```java
public boolean isValid(String s) {
    Stack<Character> stack = new Stack<>();
    Map<Character, Character> pairs = Map.of(')', '(', ']', '[', '}', '{');

    for (char c : s.toCharArray()) {
        if (pairs.containsKey(c)) {
            if (stack.isEmpty() || stack.peek() != pairs.get(c)) {
                return false;
            }
            stack.pop();
        } else {
            stack.push(c);
        }
    }

    return stack.isEmpty();
}
```

**Keywords:** "valid parentheses", "balanced brackets", "matching"
**Intuition:** Push opening brackets. Pop and match on closing brackets.

---

## Pattern 5B: Monotonic Stack - Next Greater Element

### Problem: Daily Temperatures

**Statement:** For each day, find days until warmer temperature.

```java
public int[] dailyTemperatures(int[] temperatures) {
    int n = temperatures.length;
    int[] answer = new int[n];
    Stack<Integer> stack = new Stack<>();  // Indices

    for (int i = 0; i < n; i++) {
        while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
            int prevDay = stack.pop();
            answer[prevDay] = i - prevDay;
        }
        stack.push(i);
    }

    return answer;
}
```

**Keywords:** "next greater", "next warmer", "days until"
**Intuition:** Stack holds unresolved elements. Pop when current element resolves them.

---

## Pattern 5C: Monotonic Stack - Largest Rectangle

### Problem: Largest Rectangle in Histogram

**Statement:** Find largest rectangle area in histogram.

```java
public int largestRectangleArea(int[] heights) {
    Stack<Integer> stack = new Stack<>();
    int maxArea = 0;

    for (int i = 0; i <= heights.length; i++) {
        int h = (i == heights.length) ? 0 : heights[i];

        while (!stack.isEmpty() && h < heights[stack.peek()]) {
            int height = heights[stack.pop()];
            int width = stack.isEmpty() ? i : i - stack.peek() - 1;
            maxArea = Math.max(maxArea, height * width);
        }

        stack.push(i);
    }

    return maxArea;
}
```

**Keywords:** "largest rectangle", "histogram", "maximum area"
**Intuition:** For each bar, find how far left and right it can extend. Monotonic stack finds boundaries.

---

## Pattern 5D: Min Stack (Design with O(1) Operations)

### Problem: Min Stack

**Statement:** Design stack supporting push, pop, top, and getMin in O(1).

```java
class MinStack {
    private Stack<Integer> stack;
    private Stack<Integer> minStack;

    public MinStack() {
        stack = new Stack<>();
        minStack = new Stack<>();
    }

    public void push(int val) {
        stack.push(val);
        if (minStack.isEmpty() || val <= minStack.peek()) {
            minStack.push(val);
        }
    }

    public void pop() {
        int val = stack.pop();
        if (val == minStack.peek()) {
            minStack.pop();
        }
    }

    public int top() {
        return stack.peek();
    }

    public int getMin() {
        return minStack.peek();
    }
}
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

```java
public int orangesRotting(int[][] grid) {
    int rows = grid.length, cols = grid[0].length;
    Queue<int[]> queue = new LinkedList<>();
    int fresh = 0;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (grid[r][c] == 2) {
                queue.offer(new int[] {r, c});
            } else if (grid[r][c] == 1) {
                fresh++;
            }
        }
    }

    if (fresh == 0) return 0;

    int minutes = 0;
    int[][] directions = {{-1,0}, {0,1}, {1,0}, {0,-1}};

    while (!queue.isEmpty()) {
        int size = queue.size();
        boolean rotted = false;

        for (int i = 0; i < size; i++) {
            int[] cell = queue.poll();
            for (int[] dir : directions) {
                int nr = cell[0] + dir[0];
                int nc = cell[1] + dir[1];

                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 1) {
                    grid[nr][nc] = 2;
                    queue.offer(new int[] {nr, nc});
                    fresh--;
                    rotted = true;
                }
            }
        }

        if (rotted) minutes++;
    }

    return fresh == 0 ? minutes : -1;
}
```

**Keywords:** "spreads", "minimum time", "level by level", "simultaneous"
**Intuition:** Multi-source BFS. All sources start together. Each level = 1 time unit.

---

## Pattern 6B: Monotonic Deque - Sliding Window Maximum

### Problem: Sliding Window Maximum

**Statement:** Return max of each sliding window of size k.

```java
public int[] maxSlidingWindow(int[] nums, int k) {
    int[] result = new int[nums.length - k + 1];
    Deque<Integer> deque = new ArrayDeque<>();  // Indices

    for (int i = 0; i < nums.length; i++) {
        // Remove elements smaller than current
        while (!deque.isEmpty() && nums[deque.peekLast()] < nums[i]) {
            deque.pollLast();
        }

        deque.offerLast(i);

        // Remove elements outside window
        if (deque.peekFirst() <= i - k) {
            deque.pollFirst();
        }

        // Record result
        if (i >= k - 1) {
            result[i - k + 1] = nums[deque.peekFirst()];
        }
    }

    return result;
}
```

**Keywords:** "sliding window maximum/minimum", "max of each window"
**Intuition:** Monotonic deque keeps potential maxes in decreasing order. Front is always max.

---

<a id="section-7-linked-lists"></a>

# 7. LINKED LISTS

---

## Pattern 7A: Reverse Linked List

```java
public ListNode reverseList(ListNode head) {
    ListNode prev = null;
    ListNode curr = head;

    while (curr != null) {
        ListNode next = curr.next;
        curr.next = prev;
        prev = curr;
        curr = next;
    }

    return prev;
}
```

---

## Pattern 7B: Fast/Slow - Cycle Detection

```java
public boolean hasCycle(ListNode head) {
    ListNode slow = head, fast = head;

    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
        if (slow == fast) return true;
    }

    return false;
}
```

---

## Pattern 7C: Fast/Slow - Find Middle

```java
public ListNode middleNode(ListNode head) {
    ListNode slow = head, fast = head;

    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
    }

    return slow;
}
```

---

## Pattern 7D: Two Pointers with Gap

### Problem: Remove Nth Node From End

```java
public ListNode removeNthFromEnd(ListNode head, int n) {
    ListNode dummy = new ListNode(0, head);
    ListNode first = dummy, second = dummy;

    for (int i = 0; i <= n; i++) {
        first = first.next;
    }

    while (first != null) {
        first = first.next;
        second = second.next;
    }

    second.next = second.next.next;

    return dummy.next;
}
```

---

## Pattern 7E: Merge Two Sorted Lists

```java
public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
    ListNode dummy = new ListNode(0);
    ListNode curr = dummy;

    while (list1 != null && list2 != null) {
        if (list1.val <= list2.val) {
            curr.next = list1;
            list1 = list1.next;
        } else {
            curr.next = list2;
            list2 = list2.next;
        }
        curr = curr.next;
    }

    curr.next = (list1 != null) ? list1 : list2;

    return dummy.next;
}
```

---

<a id="section-8-trees-dfs"></a>

# 8. TREES - DFS

---

## Pattern 8A: Basic DFS (Max Depth)

```java
public int maxDepth(TreeNode root) {
    if (root == null) return 0;
    return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
}
```

---

## Pattern 8B: DFS with Validation (BST)

```java
public boolean isValidBST(TreeNode root) {
    return validate(root, Long.MIN_VALUE, Long.MAX_VALUE);
}

private boolean validate(TreeNode node, long min, long max) {
    if (node == null) return true;

    if (node.val <= min || node.val >= max) return false;

    return validate(node.left, min, node.val) &&
           validate(node.right, node.val, max);
}
```

---

## Pattern 8C: DFS - Lowest Common Ancestor

```java
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null || root == p || root == q) return root;

    TreeNode left = lowestCommonAncestor(root.left, p, q);
    TreeNode right = lowestCommonAncestor(root.right, p, q);

    if (left != null && right != null) return root;

    return left != null ? left : right;
}
```

---

## Pattern 8D: DFS - Path Sum

```java
public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
    List<List<Integer>> result = new ArrayList<>();
    dfs(root, targetSum, new ArrayList<>(), result);
    return result;
}

private void dfs(TreeNode node, int remaining, List<Integer> path, List<List<Integer>> result) {
    if (node == null) return;

    path.add(node.val);

    if (node.left == null && node.right == null && remaining == node.val) {
        result.add(new ArrayList<>(path));
    }

    dfs(node.left, remaining - node.val, path, result);
    dfs(node.right, remaining - node.val, path, result);

    path.remove(path.size() - 1);
}
```

---

<a id="section-9-trees-bfs"></a>

# 9. TREES - BFS

---

## Pattern 9A: Level Order Traversal

```java
public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> result = new ArrayList<>();
    if (root == null) return result;

    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);

    while (!queue.isEmpty()) {
        int size = queue.size();
        List<Integer> level = new ArrayList<>();

        for (int i = 0; i < size; i++) {
            TreeNode node = queue.poll();
            level.add(node.val);
            if (node.left != null) queue.offer(node.left);
            if (node.right != null) queue.offer(node.right);
        }

        result.add(level);
    }

    return result;
}
```

---

## Pattern 9B: Right Side View

```java
public List<Integer> rightSideView(TreeNode root) {
    List<Integer> result = new ArrayList<>();
    if (root == null) return result;

    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);

    while (!queue.isEmpty()) {
        int size = queue.size();

        for (int i = 0; i < size; i++) {
            TreeNode node = queue.poll();
            if (i == size - 1) result.add(node.val);
            if (node.left != null) queue.offer(node.left);
            if (node.right != null) queue.offer(node.right);
        }
    }

    return result;
}
```

---

<a id="section-10-graphs-dfs"></a>

# 10. GRAPHS - DFS

---

## Pattern 10A: Flood Fill / Connected Components

### Problem: Number of Islands

```java
public int numIslands(char[][] grid) {
    if (grid == null || grid.length == 0) return 0;

    int rows = grid.length, cols = grid[0].length;
    int count = 0;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (grid[r][c] == '1') {
                count++;
                dfs(grid, r, c);
            }
        }
    }

    return count;
}

private void dfs(char[][] grid, int r, int c) {
    if (r < 0 || r >= grid.length || c < 0 || c >= grid[0].length || grid[r][c] == '0') {
        return;
    }

    grid[r][c] = '0';
    dfs(grid, r + 1, c);
    dfs(grid, r - 1, c);
    dfs(grid, r, c + 1);
    dfs(grid, r, c - 1);
}
```

---

## Pattern 10B: Clone Graph

```java
public Node cloneGraph(Node node) {
    if (node == null) return null;

    Map<Node, Node> cloned = new HashMap<>();
    return dfs(node, cloned);
}

private Node dfs(Node node, Map<Node, Node> cloned) {
    if (cloned.containsKey(node)) {
        return cloned.get(node);
    }

    Node copy = new Node(node.val);
    cloned.put(node, copy);

    for (Node neighbor : node.neighbors) {
        copy.neighbors.add(dfs(neighbor, cloned));
    }

    return copy;
}
```

---

<a id="section-11-graphs-bfs-shortest-path"></a>

# 11. GRAPHS - BFS (SHORTEST PATH)

---

## Pattern 11A: Shortest Path in Unweighted Graph

```java
public int shortestPathBinaryMatrix(int[][] grid) {
    int n = grid.length;
    if (grid[0][0] == 1 || grid[n-1][n-1] == 1) return -1;

    int[][] directions = {{-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}};
    Queue<int[]> queue = new LinkedList<>();
    queue.offer(new int[] {0, 0, 1});
    grid[0][0] = 1;

    while (!queue.isEmpty()) {
        int[] curr = queue.poll();
        int r = curr[0], c = curr[1], dist = curr[2];

        if (r == n - 1 && c == n - 1) return dist;

        for (int[] dir : directions) {
            int nr = r + dir[0], nc = c + dir[1];
            if (nr >= 0 && nr < n && nc >= 0 && nc < n && grid[nr][nc] == 0) {
                grid[nr][nc] = 1;
                queue.offer(new int[] {nr, nc, dist + 1});
            }
        }
    }

    return -1;
}
```

---

## Pattern 11B: Word Ladder

```java
public int ladderLength(String beginWord, String endWord, List<String> wordList) {
    Set<String> wordSet = new HashSet<>(wordList);
    if (!wordSet.contains(endWord)) return 0;

    Queue<String> queue = new LinkedList<>();
    queue.offer(beginWord);
    Set<String> visited = new HashSet<>();
    visited.add(beginWord);
    int level = 1;

    while (!queue.isEmpty()) {
        int size = queue.size();

        for (int i = 0; i < size; i++) {
            String word = queue.poll();

            if (word.equals(endWord)) return level;

            char[] chars = word.toCharArray();
            for (int j = 0; j < chars.length; j++) {
                char original = chars[j];
                for (char c = 'a'; c <= 'z'; c++) {
                    chars[j] = c;
                    String newWord = new String(chars);
                    if (wordSet.contains(newWord) && !visited.contains(newWord)) {
                        visited.add(newWord);
                        queue.offer(newWord);
                    }
                }
                chars[j] = original;
            }
        }

        level++;
    }

    return 0;
}
```

---

<a id="section-12-topological-sort"></a>

# 12. TOPOLOGICAL SORT

---

## Pattern 12A: Course Schedule (Cycle Detection)

```java
public boolean canFinish(int numCourses, int[][] prerequisites) {
    List<List<Integer>> graph = new ArrayList<>();
    int[] indegree = new int[numCourses];

    for (int i = 0; i < numCourses; i++) {
        graph.add(new ArrayList<>());
    }

    for (int[] prereq : prerequisites) {
        graph.get(prereq[1]).add(prereq[0]);
        indegree[prereq[0]]++;
    }

    Queue<Integer> queue = new LinkedList<>();
    for (int i = 0; i < numCourses; i++) {
        if (indegree[i] == 0) queue.offer(i);
    }

    int completed = 0;
    while (!queue.isEmpty()) {
        int course = queue.poll();
        completed++;

        for (int next : graph.get(course)) {
            indegree[next]--;
            if (indegree[next] == 0) queue.offer(next);
        }
    }

    return completed == numCourses;
}
```

---

## Pattern 12B: Course Schedule II (Return Order)

```java
public int[] findOrder(int numCourses, int[][] prerequisites) {
    List<List<Integer>> graph = new ArrayList<>();
    int[] indegree = new int[numCourses];

    for (int i = 0; i < numCourses; i++) {
        graph.add(new ArrayList<>());
    }

    for (int[] prereq : prerequisites) {
        graph.get(prereq[1]).add(prereq[0]);
        indegree[prereq[0]]++;
    }

    Queue<Integer> queue = new LinkedList<>();
    for (int i = 0; i < numCourses; i++) {
        if (indegree[i] == 0) queue.offer(i);
    }

    int[] order = new int[numCourses];
    int idx = 0;

    while (!queue.isEmpty()) {
        int course = queue.poll();
        order[idx++] = course;

        for (int next : graph.get(course)) {
            indegree[next]--;
            if (indegree[next] == 0) queue.offer(next);
        }
    }

    return idx == numCourses ? order : new int[0];
}
```

---

<a id="section-13-heaps-priority-queue"></a>

# 13. HEAPS / PRIORITY QUEUE

---

## Pattern 13A: Kth Largest Element

```java
public int findKthLargest(int[] nums, int k) {
    PriorityQueue<Integer> minHeap = new PriorityQueue<>();

    for (int num : nums) {
        minHeap.offer(num);
        if (minHeap.size() > k) {
            minHeap.poll();
        }
    }

    return minHeap.peek();
}
```

---

## Pattern 13B: Merge K Sorted Lists

```java
public ListNode mergeKLists(ListNode[] lists) {
    PriorityQueue<ListNode> heap = new PriorityQueue<>((a, b) -> a.val - b.val);

    for (ListNode head : lists) {
        if (head != null) heap.offer(head);
    }

    ListNode dummy = new ListNode(0);
    ListNode curr = dummy;

    while (!heap.isEmpty()) {
        ListNode node = heap.poll();
        curr.next = node;
        curr = curr.next;

        if (node.next != null) {
            heap.offer(node.next);
        }
    }

    return dummy.next;
}
```

---

## Pattern 13C: Find Median from Data Stream (Two Heaps)

```java
class MedianFinder {
    private PriorityQueue<Integer> small;  // Max heap
    private PriorityQueue<Integer> large;  // Min heap

    public MedianFinder() {
        small = new PriorityQueue<>(Collections.reverseOrder());
        large = new PriorityQueue<>();
    }

    public void addNum(int num) {
        small.offer(num);

        // Balance: max of small <= min of large
        if (!small.isEmpty() && !large.isEmpty() && small.peek() > large.peek()) {
            large.offer(small.poll());
        }

        // Balance sizes
        if (small.size() > large.size() + 1) {
            large.offer(small.poll());
        }
        if (large.size() > small.size()) {
            small.offer(large.poll());
        }
    }

    public double findMedian() {
        if (small.size() > large.size()) {
            return small.peek();
        }
        return (small.peek() + large.peek()) / 2.0;
    }
}
```

---

<a id="section-14-backtracking"></a>

# 14. BACKTRACKING

---

## Pattern 14A: Subsets

```java
public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    backtrack(nums, 0, new ArrayList<>(), result);
    return result;
}

private void backtrack(int[] nums, int start, List<Integer> current, List<List<Integer>> result) {
    result.add(new ArrayList<>(current));

    for (int i = start; i < nums.length; i++) {
        current.add(nums[i]);
        backtrack(nums, i + 1, current, result);
        current.remove(current.size() - 1);
    }
}
```

---

## Pattern 14B: Permutations

```java
public List<List<Integer>> permute(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    boolean[] used = new boolean[nums.length];
    backtrack(nums, new ArrayList<>(), used, result);
    return result;
}

private void backtrack(int[] nums, List<Integer> current, boolean[] used, List<List<Integer>> result) {
    if (current.size() == nums.length) {
        result.add(new ArrayList<>(current));
        return;
    }

    for (int i = 0; i < nums.length; i++) {
        if (used[i]) continue;

        used[i] = true;
        current.add(nums[i]);
        backtrack(nums, current, used, result);
        current.remove(current.size() - 1);
        used[i] = false;
    }
}
```

---

## Pattern 14C: Combination Sum

```java
public List<List<Integer>> combinationSum(int[] candidates, int target) {
    List<List<Integer>> result = new ArrayList<>();
    backtrack(candidates, target, 0, new ArrayList<>(), result);
    return result;
}

private void backtrack(int[] candidates, int remaining, int start,
                       List<Integer> current, List<List<Integer>> result) {
    if (remaining == 0) {
        result.add(new ArrayList<>(current));
        return;
    }
    if (remaining < 0) return;

    for (int i = start; i < candidates.length; i++) {
        current.add(candidates[i]);
        backtrack(candidates, remaining - candidates[i], i, current, result);
        current.remove(current.size() - 1);
    }
}
```

---

## Pattern 14D: Word Search

```java
public boolean exist(char[][] board, String word) {
    int rows = board.length, cols = board[0].length;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (backtrack(board, word, r, c, 0)) {
                return true;
            }
        }
    }

    return false;
}

private boolean backtrack(char[][] board, String word, int r, int c, int index) {
    if (index == word.length()) return true;

    if (r < 0 || r >= board.length || c < 0 || c >= board[0].length ||
        board[r][c] != word.charAt(index)) {
        return false;
    }

    char temp = board[r][c];
    board[r][c] = '#';

    boolean found = backtrack(board, word, r + 1, c, index + 1) ||
                    backtrack(board, word, r - 1, c, index + 1) ||
                    backtrack(board, word, r, c + 1, index + 1) ||
                    backtrack(board, word, r, c - 1, index + 1);

    board[r][c] = temp;

    return found;
}
```

---

## Pattern 14E: N-Queens

```java
public List<List<String>> solveNQueens(int n) {
    List<List<String>> result = new ArrayList<>();
    char[][] board = new char[n][n];
    for (char[] row : board) Arrays.fill(row, '.');

    Set<Integer> cols = new HashSet<>();
    Set<Integer> posDiag = new HashSet<>();
    Set<Integer> negDiag = new HashSet<>();

    backtrack(0, n, board, cols, posDiag, negDiag, result);
    return result;
}

private void backtrack(int row, int n, char[][] board, Set<Integer> cols,
                       Set<Integer> posDiag, Set<Integer> negDiag,
                       List<List<String>> result) {
    if (row == n) {
        List<String> solution = new ArrayList<>();
        for (char[] r : board) solution.add(new String(r));
        result.add(solution);
        return;
    }

    for (int col = 0; col < n; col++) {
        if (cols.contains(col) || posDiag.contains(row + col) || negDiag.contains(row - col)) {
            continue;
        }

        cols.add(col);
        posDiag.add(row + col);
        negDiag.add(row - col);
        board[row][col] = 'Q';

        backtrack(row + 1, n, board, cols, posDiag, negDiag, result);

        cols.remove(col);
        posDiag.remove(row + col);
        negDiag.remove(row - col);
        board[row][col] = '.';
    }
}
```

---

<a id="section-15-advanced-graph-patterns"></a>

# 15. ADVANCED GRAPH PATTERNS

| Signal | Algorithm | Complexity |
| ------ | --------- | ---------- |
| Unweighted shortest path | BFS | O(V + E) |
| Many sources start together | Multi-source BFS | O(V + E) |
| Weights are only 0 or 1 | 0-1 BFS | O(V + E) |
| Non-negative weights | Dijkstra | O((V + E) log V) |
| Negative edges or at most K edges | Bellman-Ford | O(VE), or O(KE) |
| All-pairs distances, small graph | Floyd-Warshall | O(V^3) |
| Repeated merges/connectivity | Union-Find | Almost O(1) per operation |
| Minimum total connection cost | Kruskal / Prim MST | O(E log E) / O(E log V) |
| Critical edges / strongly connected regions | Low-link DFS | O(V + E) |
| Use every edge exactly once | Hierholzer | O(E) plus edge ordering |
| Position plus keys/stops/resources | BFS/Dijkstra on full state | Depends on state count |

---

## Pattern 15A: Dijkstra - Non-Negative Weighted Shortest Path

### Problem: Network Delay Time

```java
public int networkDelayTime(int[][] times, int n, int start) {
    List<int[]>[] graph = new ArrayList[n + 1];
    for (int node = 1; node <= n; node++) graph[node] = new ArrayList<>();
    for (int[] edge : times) graph[edge[0]].add(new int[]{edge[1], edge[2]});

    long[] distance = new long[n + 1];
    Arrays.fill(distance, Long.MAX_VALUE);
    distance[start] = 0;

    PriorityQueue<long[]> minHeap =
        new PriorityQueue<>(Comparator.comparingLong(entry -> entry[0]));
    minHeap.offer(new long[]{0, start}); // distance, node

    while (!minHeap.isEmpty()) {
        long[] current = minHeap.poll();
        long currentDistance = current[0];
        int node = (int) current[1];
        if (currentDistance != distance[node]) continue; // stale entry

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

**Invariant:** with non-negative weights, removing the current best distance from the heap finalises it. Do not mark a node visited when first inserting it; a cheaper entry may arrive later.

---

## Pattern 15B: 0-1 BFS - Binary Edge Weights

### Problem: Minimum Obstacle Removal to Reach Corner

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
        int[] cell = deque.pollFirst();
        int row = cell[0], col = cell[1];

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

**Keywords:** "0/1 cost", "free or paid move", "minimum reversals/obstacles". Zero-cost edges go to the front and cost-one edges to the back, giving Dijkstra's ordering in O(V + E).

---

## Pattern 15C: Bellman-Ford by Layers - Bounded Stops

### Problem: Cheapest Flights Within K Stops

```java
public int findCheapestPrice(int n, int[][] flights, int source, int target, int k) {
    long infinity = Long.MAX_VALUE / 4;
    long[] cost = new long[n];
    Arrays.fill(cost, infinity);
    cost[source] = 0;

    // At most k stops means at most k + 1 edges.
    for (int edgesUsed = 0; edgesUsed <= k; edgesUsed++) {
        long[] nextCost = cost.clone();
        for (int[] flight : flights) {
            int from = flight[0], to = flight[1], price = flight[2];
            if (cost[from] != infinity) {
                nextCost[to] = Math.min(nextCost[to], cost[from] + price);
            }
        }
        cost = nextCost;
    }

    return cost[target] == infinity ? -1 : (int) cost[target];
}
```

**Critical detail:** read from `cost` and write to the cloned layer. In-place relaxation can chain several edges in one round and break the stop limit.

---

## Pattern 15D: Union-Find - Dynamic Connectivity and Cycle Detection

### Problem: Redundant Connection

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

**Keywords:** "merge groups", "redundant edge", "dynamic connectivity". An edge whose endpoints already have the same root closes an undirected cycle.

---

## Pattern 15E: Prim's Minimum Spanning Tree

### Problem: Min Cost to Connect All Points

```java
public int minCostConnectPoints(int[][] points) {
    int numberOfPoints = points.length;
    boolean[] inTree = new boolean[numberOfPoints];
    PriorityQueue<int[]> minHeap =
        new PriorityQueue<>(Comparator.comparingInt(edge -> edge[0]));
    minHeap.offer(new int[]{0, 0}); // cost, destination

    int totalCost = 0, used = 0;
    while (used < numberOfPoints) {
        int[] edge = minHeap.poll();
        int cost = edge[0], point = edge[1];
        if (inTree[point]) continue;

        inTree[point] = true;
        totalCost += cost;
        used++;

        for (int neighbour = 0; neighbour < numberOfPoints; neighbour++) {
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

**MST versus shortest path:** MST minimises the total chosen network cost; Dijkstra minimises routes from one source. Standard adjacency-list Prim is O(E log V); this implicit complete graph version is O(V^2 log V).

---

## Pattern 15F: Bipartite Graph - Two-Colouring

### Problem: Is Graph Bipartite?

```java
public boolean isBipartite(int[][] graph) {
    int[] colour = new int[graph.length];
    Arrays.fill(colour, -1);

    for (int start = 0; start < graph.length; start++) {
        if (colour[start] != -1) continue;
        colour[start] = 0;
        Queue<Integer> queue = new ArrayDeque<>();
        queue.offer(start);

        while (!queue.isEmpty()) {
            int node = queue.poll();
            for (int neighbour : graph[node]) {
                if (colour[neighbour] == -1) {
                    colour[neighbour] = colour[node] ^ 1;
                    queue.offer(neighbour);
                } else if (colour[neighbour] == colour[node]) {
                    return false;
                }
            }
        }
    }
    return true;
}
```

**Invariant:** each edge must cross between colours. Checking every start node handles disconnected graphs; a conflict proves an odd cycle.

---

## Pattern 15G: Tarjan Low-Link - Bridges

### Problem: Critical Connections in a Network

```java
class BridgeFinder {
    private List<int[]>[] graph;
    private int[] discovery;
    private int[] low;
    private int timer;
    private final List<List<Integer>> bridges = new ArrayList<>();

    List<List<Integer>> criticalConnections(int n, List<List<Integer>> connections) {
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

**Meaning of `low[v]`:** earliest discovery reachable from `v`'s subtree without its parent edge. `low[v] > discovery[u]` means `u-v` has no alternate route. Tracking parent edge ID also handles parallel edges correctly.

---

## Pattern 15H: State-Space BFS - Node Plus Bitmask

### Problem: Shortest Path to Get All Keys

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

**The hard-problem unlock:** the vertex is `(row, col, keys)`, not merely `(row, col)`. Different key masks at the same cell allow different future moves.

---

## Pattern 15I: Eulerian Path - Use Every Edge Once

### Problem: Reconstruct Itinerary

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

**Keywords:** "use every ticket/edge exactly once", "arrange pairs", "valid itinerary". Hierholzer appends when stuck, so the route is built in reverse. Eulerian uses every edge once; Hamiltonian uses every vertex once.

---

## Advanced Graph Interview Checklist

1. Define the real vertex and edge, including keys/stops/resources.
2. State whether edges are directed, weighted, and possibly negative.
3. Match weights to BFS, 0-1 BFS, Dijkstra, Bellman-Ford, or DAG relaxation.
4. Cover every component when no single source is guaranteed.
5. Skip stale Dijkstra heap entries; mark ordinary BFS states on enqueue.
6. Distinguish MST total cost from single-source shortest paths.
7. Prefer iterative DFS if a chain can overflow Java's call stack.
8. Give complexity in V and E—or in the expanded number of states.

**Next drills:** Accounts Merge, Swim in Rising Water, Alien Dictionary, Network Delay Time, Min Cost to Connect All Points, Critical Connections, Reconstruct Itinerary, and Shortest Path to Get All Keys.

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

---

<a id="section-java-syntax-cheat-sheet"></a>

# Java Syntax Cheat Sheet

```java
// HashMap
Map<K, V> map = new HashMap<>();
map.put(key, value);
map.get(key);
map.getOrDefault(key, defaultValue);
map.containsKey(key);
map.computeIfAbsent(key, k -> new ArrayList<>());

// HashSet
Set<T> set = new HashSet<>();
set.add(value);
set.contains(value);
set.remove(value);

// ArrayList
List<T> list = new ArrayList<>();
list.add(value);
list.get(index);
list.set(index, value);
list.remove(list.size() - 1);

// Queue
Queue<T> queue = new LinkedList<>();
queue.offer(value);
queue.poll();
queue.peek();
queue.isEmpty();

// Deque
Deque<T> deque = new ArrayDeque<>();
deque.offerFirst(value);
deque.offerLast(value);
deque.pollFirst();
deque.pollLast();
deque.peekFirst();
deque.peekLast();

// Stack (prefer Deque)
Deque<T> stack = new ArrayDeque<>();
stack.push(value);
stack.pop();
stack.peek();

// PriorityQueue
PriorityQueue<T> minHeap = new PriorityQueue<>();
PriorityQueue<T> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[0] - b[0]);
pq.offer(value);
pq.poll();
pq.peek();

// Arrays
Arrays.sort(arr);
Arrays.fill(arr, value);
Arrays.copyOf(arr, length);
Arrays.asList(1, 2, 3);

// String
s.charAt(i);
s.substring(start, end);
s.toCharArray();
String.valueOf(num);
Integer.parseInt(s);

// StringBuilder
StringBuilder sb = new StringBuilder();
sb.append("text");
sb.toString();
sb.setCharAt(index, char);
```
