# Complete DSA Patterns: Multiple Problems Per Pattern (C++)

<a id="quick-navigation"></a>

## Quick navigation

- [0. C++ LANGUAGE & DATA STRUCTURE CHEAT SHEET](#section-0-c-plus-plus-language-and-data-structure-cheat-sheet)
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
- [C++ Syntax Cheat Sheet](#section-c-plus-plus-syntax-cheat-sheet)


---

<a id="section-0-c-plus-plus-language-and-data-structure-cheat-sheet"></a>

# 0. C++ LANGUAGE & DATA STRUCTURE CHEAT SHEET

## Imports, aliases, and basic syntax

```cpp
#include <algorithm>
#include <array>
#include <climits>
#include <deque>
#include <functional>
#include <numeric>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;

long long solve(const vector<int>& nums) { // const reference avoids a copy
    long long answer = 0;
    return answer;
}

swap(left, right);
const long long INF = LLONG_MAX / 4;
```

## Loops

```cpp
for (int index = 0; index < static_cast<int>(nums.size()); ++index) {
    int value = nums[index];
}

for (int index = static_cast<int>(nums.size()) - 1; index >= 0; --index) {
    int value = nums[index];
}

for (int value : nums) {          // copy each value
}

for (int& value : nums) {         // mutate each value
    value *= 2;
}

for (const int& value : nums) {   // no copy, no mutation
}

for (auto [key, value] : mapping) {
}

for (auto iterator = seen.begin(); iterator != seen.end(); ) {
    if (shouldRemove(*iterator)) iterator = seen.erase(iterator);
    else ++iterator;
}

while (left <= right) {
    ++left;
}
// break exits one loop; continue skips to its next iteration.
```

## Arrays and vectors

```cpp
int fixed[3] = {3, 1, 2};
array<int, 3> saferFixed = {3, 1, 2};

vector<int> nums = {3, 1, 2};
vector<int> zeros(n, 0);
vector<vector<int>> matrix(rows, vector<int>(cols, 0));

nums.push_back(4);                    // append
nums.emplace_back(5);                 // construct at end
nums.insert(nums.begin() + index, 99);
nums.insert(nums.end(), other.begin(), other.end());

int last = nums.back();
nums.pop_back();                      // removes; returns void
nums.erase(nums.begin() + index);
nums.erase(nums.begin() + from, nums.begin() + to);
nums.clear();

int value = nums[index];              // unchecked
int checked = nums.at(index);         // throws if out of bounds
bool empty = nums.empty();
int size = nums.size();
bool exists = find(nums.begin(), nums.end(), target) != nums.end();

sort(nums.begin(), nums.end());
sort(nums.rbegin(), nums.rend());
reverse(nums.begin(), nums.end());
fill(nums.begin(), nums.end(), -1);

vector<int> copy = nums;
vector<int> slice(nums.begin() + from, nums.begin() + to);

// Remove all matching values: erase-remove idiom
nums.erase(remove(nums.begin(), nums.end(), target), nums.end());
```

`vector<bool>` is bit-packed and returns proxy references; use `vector<char>` when normal reference behaviour matters.

## Strings

```cpp
string text = "graph";
int length = text.size();
char character = text[index];
string substring = text.substr(start, count);
bool contains = text.find("rap") != string::npos;
size_t position = text.find('a');             // string::npos if absent

text.push_back('!');
text += "s";
text.insert(index, "X");
text.erase(index, count);
char last = text.back();
text.pop_back();
reverse(text.begin(), text.end());

int number = stoi("42");
long long large = stoll("42");
string digits = to_string(42);
```

Strings are mutable in C++, but insert/erase in the middle is O(n).

## Hash maps and ordered maps

```cpp
unordered_map<int, int> mapping;
mapping[key] = value;                    // inserts default if key is absent
mapping.insert({key, value});            // does not replace existing key
mapping.emplace(key, value);
mapping[key]++;

bool exists = mapping.find(key) != mapping.end();
int value = mapping.at(key);             // throws if absent
int safe = mapping.find(key) == mapping.end() ? fallback : mapping.at(key);

int removedCount = mapping.erase(key);   // 0 or 1
mapping.clear();

for (const auto& [key, value] : mapping) {
}

unordered_map<int, vector<int>> groups;
groups[key].push_back(value);

map<int, int> ordered;                   // sorted keys, O(log n)
auto firstAtLeast = ordered.lower_bound(target);
auto firstGreater = ordered.upper_bound(target);
```

Average `unordered_map` operations are O(1); `map` operations are O(log n). Since these guides target C++17 too, prefer `find(...) != end()` over `contains`.

## Sets

```cpp
unordered_set<int> seen;
seen.insert(value);
seen.emplace(value);
bool exists = seen.find(value) != seen.end();
int removedCount = seen.erase(value);
seen.clear();

set<int> ordered;                        // unique + sorted
ordered.insert(value);
auto smallest = ordered.begin();
auto largest = prev(ordered.end());
auto firstAtLeast = ordered.lower_bound(target);
```

## Stack, queue, and deque

```cpp
stack<int> stackValues;
stackValues.push(value);
int top = stackValues.top();
stackValues.pop();
bool stackEmpty = stackValues.empty();

queue<int> queueValues;
queueValues.push(value);
int front = queueValues.front();
int back = queueValues.back();
queueValues.pop();
bool queueEmpty = queueValues.empty();

deque<int> dequeValues;
dequeValues.push_front(value);
dequeValues.push_back(value);
int first = dequeValues.front();
int last = dequeValues.back();
dequeValues.pop_front();
dequeValues.pop_back();
```

C++ `pop()` operations return `void`; read `top()`, `front()`, or `back()` before removing.

## Heap / priority_queue

```cpp
priority_queue<int> maxHeap;             // max-heap by default
maxHeap.push(value);
int largest = maxHeap.top();
maxHeap.pop();

priority_queue<int, vector<int>, greater<int>> minHeap;
minHeap.push(value);
int smallest = minHeap.top();

using State = pair<long long, int>;       // distance, node
priority_queue<State, vector<State>, greater<State>> byDistance;
byDistance.push({distance, node});
auto [bestDistance, nextNode] = byDistance.top();
byDistance.pop();

auto compare = [](const vector<int>& left, const vector<int>& right) {
    return left[0] > right[0];            // greater priority value sinks
};
priority_queue<vector<int>, vector<vector<int>>, decltype(compare)> custom(compare);
```

Push/pop is O(log n); `top()` is O(1).

## Pairs, tuples, and structured bindings

```cpp
pair<int, int> point = {row, col};
int row = point.first;
int col = point.second;
auto [r, c] = point;

tuple<int, int, int> state = {row, col, mask};
auto [stateRow, stateCol, mask] = state;
int first = get<0>(state);
```

Pairs and tuples compare lexicographically, which is useful in heaps and sorting.

## Algorithms and binary search

```cpp
sort(nums.begin(), nums.end());
auto found = find(nums.begin(), nums.end(), target);
int total = accumulate(nums.begin(), nums.end(), 0);
long long largeTotal = accumulate(nums.begin(), nums.end(), 0LL);
int minimum = *min_element(nums.begin(), nums.end());
int maximum = *max_element(nums.begin(), nums.end());

bool exists = binary_search(nums.begin(), nums.end(), target);
auto firstAtLeast = lower_bound(nums.begin(), nums.end(), target);
auto firstGreater = upper_bound(nums.begin(), nums.end(), target);
int index = lower_bound(nums.begin(), nums.end(), target) - nums.begin();

sort(pairs.begin(), pairs.end(), [](const auto& left, const auto& right) {
    if (left.second != right.second) return left.second < right.second;
    return left.first < right.first;
});
```

Binary-search algorithms require sorted input.

## Linked structures and graphs

```cpp
struct ListNode {
    int val;
    ListNode* next;
    explicit ListNode(int value) : val(value), next(nullptr) {}
};

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(int value) : val(value), left(nullptr), right(nullptr) {}
};

vector<vector<int>> graph(n);
for (const auto& edge : edges) {
    int first = edge[0], second = edge[1];
    graph[first].push_back(second);
    graph[second].push_back(first);        // omit for directed graph
}

vector<vector<pair<int, int>>> weighted(n);
weighted[source].push_back({target, weight});

vector<char> visited(n, false);
int directions4[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
```

## Interview traps

- Use `long long` for accumulated sums, products, distances, and comparator keys.
- Avoid unsigned underflow: `size() - 1` is dangerous when a container is empty; cast after checking or use reverse iterators.
- Inserting into a vector can invalidate iterators, pointers, and references when it reallocates.
- `unordered_map[key]` mutates the map when the key is missing; use `find` for a read-only check.
- Pass large inputs as `const T&`; pass as `T&` only when mutation is intended.
- Prefer RAII containers and smart pointers to raw `new`/`delete` in interview code.

## Complexity reminders

| Structure | Add | Remove | Membership / lookup | Peek |
| --------- | --- | ------ | ------------------- | ---- |
| `vector` end | O(1)* | O(1) | O(n), index O(1) | O(1) |
| `vector` middle | O(n) | O(n) | O(n) | O(1) |
| `unordered_map/set` average | O(1) | O(1) | O(1) | — |
| `map/set` | O(log n) | O(log n) | O(log n) | O(log n) |
| `deque` ends | O(1) | O(1) | O(n), index O(1) | O(1) |
| `priority_queue` | O(log n) | O(log n) | O(n) | O(1) |

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

```cpp
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> seen;  // value -> index

    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];
        if (seen.count(complement)) {
            return {seen[complement], i};
        }
        seen[nums[i]] = i;
    }

    return {};
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

```cpp
vector<int> topKFrequent(vector<int>& nums, int k) {
    // Count frequencies
    unordered_map<int, int> count;
    for (int num : nums) {
        count[num]++;
    }

    // Bucket sort: index = frequency
    vector<vector<int>> buckets(nums.size() + 1);

    for (auto& [num, freq] : count) {
        buckets[freq].push_back(num);
    }

    // Collect from highest frequency
    vector<int> result;
    for (int freq = buckets.size() - 1; freq >= 0 && result.size() < k; freq--) {
        for (int num : buckets[freq]) {
            result.push_back(num);
            if (result.size() == k) break;
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

```cpp
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, vector<string>> groups;

    for (const string& s : strs) {
        string key = s;
        sort(key.begin(), key.end());
        groups[key].push_back(s);
    }

    vector<vector<string>> result;
    for (auto& [key, group] : groups) {
        result.push_back(group);
    }

    return result;
}
```

**Alternative Key (character count):**

```cpp
vector<vector<string>> groupAnagramsV2(vector<string>& strs) {
    unordered_map<string, vector<string>> groups;

    for (const string& s : strs) {
        string key(26, '0');
        for (char c : s) {
            key[c - 'a']++;
        }
        groups[key].push_back(s);
    }

    vector<vector<string>> result;
    for (auto& [key, group] : groups) {
        result.push_back(group);
    }

    return result;
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

```cpp
int subarraySum(vector<int>& nums, int k) {
    int count = 0;
    int prefixSum = 0;
    unordered_map<int, int> seen;
    seen[0] = 1;  // prefix_sum -> how many times seen

    for (int num : nums) {
        prefixSum += num;

        // If (prefixSum - k) was seen before, those subarrays sum to k
        if (seen.count(prefixSum - k)) {
            count += seen[prefixSum - k];
        }

        seen[prefixSum]++;
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

```cpp
int maxArea(vector<int>& height) {
    int left = 0, right = height.size() - 1;
    int maxWater = 0;

    while (left < right) {
        int width = right - left;
        int h = min(height[left], height[right]);
        maxWater = max(maxWater, width * h);

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

```cpp
vector<vector<int>> threeSum(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> result;

    for (int i = 0; i < (int)nums.size() - 2; i++) {
        // Skip duplicates
        if (i > 0 && nums[i] == nums[i - 1]) continue;

        int left = i + 1, right = nums.size() - 1;
        int target = -nums[i];

        while (left < right) {
            int sum = nums[left] + nums[right];

            if (sum == target) {
                result.push_back({nums[i], nums[left], nums[right]});
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

```cpp
int removeDuplicates(vector<int>& nums) {
    if (nums.empty()) return 0;

    int slow = 0;

    for (int fast = 1; fast < nums.size(); fast++) {
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

```cpp
void moveZeroes(vector<int>& nums) {
    int slow = 0;

    for (int fast = 0; fast < nums.size(); fast++) {
        if (nums[fast] != 0) {
            swap(nums[slow], nums[fast]);
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

```cpp
int maxSumSubarray(vector<int>& nums, int k) {
    if (nums.size() < k) return 0;

    // Calculate first window
    int windowSum = 0;
    for (int i = 0; i < k; i++) {
        windowSum += nums[i];
    }
    int maxSum = windowSum;

    // Slide the window
    for (int i = k; i < nums.size(); i++) {
        windowSum += nums[i] - nums[i - k];
        maxSum = max(maxSum, windowSum);
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

```cpp
int lengthOfLongestSubstring(string s) {
    unordered_set<char> window;
    int left = 0;
    int maxLength = 0;

    for (int right = 0; right < s.size(); right++) {
        while (window.count(s[right])) {
            window.erase(s[left]);
            left++;
        }

        window.insert(s[right]);
        maxLength = max(maxLength, right - left + 1);
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

```cpp
int characterReplacement(string s, int k) {
    vector<int> count(26, 0);
    int left = 0;
    int maxFreq = 0;
    int maxLength = 0;

    for (int right = 0; right < s.size(); right++) {
        count[s[right] - 'A']++;
        maxFreq = max(maxFreq, count[s[right] - 'A']);

        // Window size - maxFreq = chars to replace
        while ((right - left + 1) - maxFreq > k) {
            count[s[left] - 'A']--;
            left++;
        }

        maxLength = max(maxLength, right - left + 1);
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

```cpp
string minWindow(string s, string t) {
    if (s.empty() || t.empty()) return "";

    unordered_map<char, int> need;
    for (char c : t) {
        need[c]++;
    }

    unordered_map<char, int> have;
    int required = need.size();
    int formed = 0;

    int left = 0;
    int minLen = INT_MAX;
    int resultLeft = 0, resultRight = 0;

    for (int right = 0; right < s.size(); right++) {
        char c = s[right];
        have[c]++;

        if (need.count(c) && have[c] == need[c]) {
            formed++;
        }

        while (formed == required) {
            if (right - left + 1 < minLen) {
                minLen = right - left + 1;
                resultLeft = left;
                resultRight = right;
            }

            char leftChar = s[left];
            have[leftChar]--;
            if (need.count(leftChar) && have[leftChar] < need[leftChar]) {
                formed--;
            }
            left++;
        }
    }

    return minLen == INT_MAX ? "" : s.substr(resultLeft, minLen);
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

```cpp
int search(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;

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

```cpp
vector<int> searchRange(vector<int>& nums, int target) {
    return {findLeft(nums, target), findRight(nums, target)};
}

int findLeft(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
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

int findRight(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
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

```cpp
int searchRotated(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;

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

```cpp
int minEatingSpeed(vector<int>& piles, int h) {
    int left = 1;
    int right = *max_element(piles.begin(), piles.end());

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

bool canFinish(vector<int>& piles, int speed, int h) {
    long long hours = 0;
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

```cpp
bool isValid(string s) {
    stack<char> stk;
    unordered_map<char, char> pairs = {{')', '('}, {']', '['}, {'}', '{'}};

    for (char c : s) {
        if (pairs.count(c)) {
            if (stk.empty() || stk.top() != pairs[c]) {
                return false;
            }
            stk.pop();
        } else {
            stk.push(c);
        }
    }

    return stk.empty();
}
```

**Keywords:** "valid parentheses", "balanced brackets", "matching"
**Intuition:** Push opening brackets. Pop and match on closing brackets.

---

## Pattern 5B: Monotonic Stack - Next Greater Element

### Problem: Daily Temperatures

**Statement:** For each day, find days until warmer temperature.

```cpp
vector<int> dailyTemperatures(vector<int>& temperatures) {
    int n = temperatures.size();
    vector<int> answer(n, 0);
    stack<int> stk;  // Indices

    for (int i = 0; i < n; i++) {
        while (!stk.empty() && temperatures[i] > temperatures[stk.top()]) {
            int prevDay = stk.top();
            stk.pop();
            answer[prevDay] = i - prevDay;
        }
        stk.push(i);
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

```cpp
int largestRectangleArea(vector<int>& heights) {
    stack<int> stk;
    int maxArea = 0;

    for (int i = 0; i <= heights.size(); i++) {
        int h = (i == heights.size()) ? 0 : heights[i];

        while (!stk.empty() && h < heights[stk.top()]) {
            int height = heights[stk.top()];
            stk.pop();
            int width = stk.empty() ? i : i - stk.top() - 1;
            maxArea = max(maxArea, height * width);
        }

        stk.push(i);
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

```cpp
class MinStack {
private:
    stack<int> stk;
    stack<int> minStk;

public:
    MinStack() {}

    void push(int val) {
        stk.push(val);
        if (minStk.empty() || val <= minStk.top()) {
            minStk.push(val);
        }
    }

    void pop() {
        int val = stk.top();
        stk.pop();
        if (val == minStk.top()) {
            minStk.pop();
        }
    }

    int top() {
        return stk.top();
    }

    int getMin() {
        return minStk.top();
    }
};
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

```cpp
int orangesRotting(vector<vector<int>>& grid) {
    int rows = grid.size(), cols = grid[0].size();
    queue<pair<int, int>> q;
    int fresh = 0;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (grid[r][c] == 2) {
                q.push({r, c});
            } else if (grid[r][c] == 1) {
                fresh++;
            }
        }
    }

    if (fresh == 0) return 0;

    int minutes = 0;
    vector<pair<int, int>> directions = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};

    while (!q.empty()) {
        int size = q.size();
        bool rotted = false;

        for (int i = 0; i < size; i++) {
            auto [r, c] = q.front();
            q.pop();

            for (auto [dr, dc] : directions) {
                int nr = r + dr;
                int nc = c + dc;

                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 1) {
                    grid[nr][nc] = 2;
                    q.push({nr, nc});
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

```cpp
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    vector<int> result;
    deque<int> dq;  // Indices

    for (int i = 0; i < nums.size(); i++) {
        // Remove elements smaller than current
        while (!dq.empty() && nums[dq.back()] < nums[i]) {
            dq.pop_back();
        }

        dq.push_back(i);

        // Remove elements outside window
        if (dq.front() <= i - k) {
            dq.pop_front();
        }

        // Record result
        if (i >= k - 1) {
            result.push_back(nums[dq.front()]);
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

```cpp
ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* curr = head;

    while (curr != nullptr) {
        ListNode* next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }

    return prev;
}
```

---

## Pattern 7B: Fast/Slow - Cycle Detection

```cpp
bool hasCycle(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;

    while (fast != nullptr && fast->next != nullptr) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return true;
    }

    return false;
}
```

---

## Pattern 7C: Fast/Slow - Find Middle

```cpp
ListNode* middleNode(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;

    while (fast != nullptr && fast->next != nullptr) {
        slow = slow->next;
        fast = fast->next->next;
    }

    return slow;
}
```

---

## Pattern 7D: Two Pointers with Gap

### Problem: Remove Nth Node From End

```cpp
ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode dummy(0, head);
    ListNode* first = &dummy;
    ListNode* second = &dummy;

    for (int i = 0; i <= n; i++) {
        first = first->next;
    }

    while (first != nullptr) {
        first = first->next;
        second = second->next;
    }

    ListNode* toDelete = second->next;
    second->next = second->next->next;
    delete toDelete;

    return dummy.next;
}
```

---

## Pattern 7E: Merge Two Sorted Lists

```cpp
ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
    ListNode dummy(0);
    ListNode* curr = &dummy;

    while (list1 != nullptr && list2 != nullptr) {
        if (list1->val <= list2->val) {
            curr->next = list1;
            list1 = list1->next;
        } else {
            curr->next = list2;
            list2 = list2->next;
        }
        curr = curr->next;
    }

    curr->next = (list1 != nullptr) ? list1 : list2;

    return dummy.next;
}
```

---

<a id="section-8-trees-dfs"></a>

# 8. TREES - DFS

---

## Pattern 8A: Basic DFS (Max Depth)

```cpp
int maxDepth(TreeNode* root) {
    if (root == nullptr) return 0;
    return 1 + max(maxDepth(root->left), maxDepth(root->right));
}
```

---

## Pattern 8B: DFS with Validation (BST)

```cpp
bool isValidBST(TreeNode* root) {
    return validate(root, LONG_MIN, LONG_MAX);
}

bool validate(TreeNode* node, long minVal, long maxVal) {
    if (node == nullptr) return true;

    if (node->val <= minVal || node->val >= maxVal) return false;

    return validate(node->left, minVal, node->val) &&
           validate(node->right, node->val, maxVal);
}
```

---

## Pattern 8C: DFS - Lowest Common Ancestor

```cpp
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (root == nullptr || root == p || root == q) return root;

    TreeNode* left = lowestCommonAncestor(root->left, p, q);
    TreeNode* right = lowestCommonAncestor(root->right, p, q);

    if (left != nullptr && right != nullptr) return root;

    return left != nullptr ? left : right;
}
```

---

## Pattern 8D: DFS - Path Sum

```cpp
vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
    vector<vector<int>> result;
    vector<int> path;
    dfs(root, targetSum, path, result);
    return result;
}

void dfs(TreeNode* node, int remaining, vector<int>& path, vector<vector<int>>& result) {
    if (node == nullptr) return;

    path.push_back(node->val);

    if (node->left == nullptr && node->right == nullptr && remaining == node->val) {
        result.push_back(path);
    }

    dfs(node->left, remaining - node->val, path, result);
    dfs(node->right, remaining - node->val, path, result);

    path.pop_back();
}
```

---

<a id="section-9-trees-bfs"></a>

# 9. TREES - BFS

---

## Pattern 9A: Level Order Traversal

```cpp
vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> result;
    if (root == nullptr) return result;

    queue<TreeNode*> q;
    q.push(root);

    while (!q.empty()) {
        int size = q.size();
        vector<int> level;

        for (int i = 0; i < size; i++) {
            TreeNode* node = q.front();
            q.pop();
            level.push_back(node->val);
            if (node->left != nullptr) q.push(node->left);
            if (node->right != nullptr) q.push(node->right);
        }

        result.push_back(level);
    }

    return result;
}
```

---

## Pattern 9B: Right Side View

```cpp
vector<int> rightSideView(TreeNode* root) {
    vector<int> result;
    if (root == nullptr) return result;

    queue<TreeNode*> q;
    q.push(root);

    while (!q.empty()) {
        int size = q.size();

        for (int i = 0; i < size; i++) {
            TreeNode* node = q.front();
            q.pop();
            if (i == size - 1) result.push_back(node->val);
            if (node->left != nullptr) q.push(node->left);
            if (node->right != nullptr) q.push(node->right);
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

```cpp
int numIslands(vector<vector<char>>& grid) {
    if (grid.empty()) return 0;

    int rows = grid.size(), cols = grid[0].size();
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

void dfs(vector<vector<char>>& grid, int r, int c) {
    if (r < 0 || r >= grid.size() || c < 0 || c >= grid[0].size() || grid[r][c] == '0') {
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

```cpp
Node* cloneGraph(Node* node) {
    if (node == nullptr) return nullptr;

    unordered_map<Node*, Node*> cloned;
    return dfs(node, cloned);
}

Node* dfs(Node* node, unordered_map<Node*, Node*>& cloned) {
    if (cloned.count(node)) {
        return cloned[node];
    }

    Node* copy = new Node(node->val);
    cloned[node] = copy;

    for (Node* neighbor : node->neighbors) {
        copy->neighbors.push_back(dfs(neighbor, cloned));
    }

    return copy;
}
```

---

<a id="section-11-graphs-bfs-shortest-path"></a>

# 11. GRAPHS - BFS (SHORTEST PATH)

---

## Pattern 11A: Shortest Path in Unweighted Graph

```cpp
int shortestPathBinaryMatrix(vector<vector<int>>& grid) {
    int n = grid.size();
    if (grid[0][0] == 1 || grid[n-1][n-1] == 1) return -1;

    vector<pair<int, int>> directions = {{-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}};
    queue<tuple<int, int, int>> q;
    q.push({0, 0, 1});
    grid[0][0] = 1;

    while (!q.empty()) {
        auto [r, c, dist] = q.front();
        q.pop();

        if (r == n - 1 && c == n - 1) return dist;

        for (auto [dr, dc] : directions) {
            int nr = r + dr, nc = c + dc;
            if (nr >= 0 && nr < n && nc >= 0 && nc < n && grid[nr][nc] == 0) {
                grid[nr][nc] = 1;
                q.push({nr, nc, dist + 1});
            }
        }
    }

    return -1;
}
```

---

## Pattern 11B: Word Ladder

```cpp
int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
    unordered_set<string> wordSet(wordList.begin(), wordList.end());
    if (!wordSet.count(endWord)) return 0;

    queue<string> q;
    q.push(beginWord);
    unordered_set<string> visited;
    visited.insert(beginWord);
    int level = 1;

    while (!q.empty()) {
        int size = q.size();

        for (int i = 0; i < size; i++) {
            string word = q.front();
            q.pop();

            if (word == endWord) return level;

            for (int j = 0; j < word.size(); j++) {
                char original = word[j];
                for (char c = 'a'; c <= 'z'; c++) {
                    word[j] = c;
                    if (wordSet.count(word) && !visited.count(word)) {
                        visited.insert(word);
                        q.push(word);
                    }
                }
                word[j] = original;
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

```cpp
bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    vector<vector<int>> graph(numCourses);
    vector<int> indegree(numCourses, 0);

    for (auto& prereq : prerequisites) {
        graph[prereq[1]].push_back(prereq[0]);
        indegree[prereq[0]]++;
    }

    queue<int> q;
    for (int i = 0; i < numCourses; i++) {
        if (indegree[i] == 0) q.push(i);
    }

    int completed = 0;
    while (!q.empty()) {
        int course = q.front();
        q.pop();
        completed++;

        for (int next : graph[course]) {
            indegree[next]--;
            if (indegree[next] == 0) q.push(next);
        }
    }

    return completed == numCourses;
}
```

---

## Pattern 12B: Course Schedule II (Return Order)

```cpp
vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
    vector<vector<int>> graph(numCourses);
    vector<int> indegree(numCourses, 0);

    for (auto& prereq : prerequisites) {
        graph[prereq[1]].push_back(prereq[0]);
        indegree[prereq[0]]++;
    }

    queue<int> q;
    for (int i = 0; i < numCourses; i++) {
        if (indegree[i] == 0) q.push(i);
    }

    vector<int> order;

    while (!q.empty()) {
        int course = q.front();
        q.pop();
        order.push_back(course);

        for (int next : graph[course]) {
            indegree[next]--;
            if (indegree[next] == 0) q.push(next);
        }
    }

    return order.size() == numCourses ? order : vector<int>();
}
```

---

<a id="section-13-heaps-priority-queue"></a>

# 13. HEAPS / PRIORITY QUEUE

---

## Pattern 13A: Kth Largest Element

```cpp
int findKthLargest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> minHeap;

    for (int num : nums) {
        minHeap.push(num);
        if (minHeap.size() > k) {
            minHeap.pop();
        }
    }

    return minHeap.top();
}
```

---

## Pattern 13B: Merge K Sorted Lists

```cpp
ListNode* mergeKLists(vector<ListNode*>& lists) {
    auto cmp = [](ListNode* a, ListNode* b) { return a->val > b->val; };
    priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> heap(cmp);

    for (ListNode* head : lists) {
        if (head != nullptr) heap.push(head);
    }

    ListNode dummy(0);
    ListNode* curr = &dummy;

    while (!heap.empty()) {
        ListNode* node = heap.top();
        heap.pop();
        curr->next = node;
        curr = curr->next;

        if (node->next != nullptr) {
            heap.push(node->next);
        }
    }

    return dummy.next;
}
```

---

## Pattern 13C: Find Median from Data Stream (Two Heaps)

```cpp
class MedianFinder {
private:
    priority_queue<int> small;  // Max heap
    priority_queue<int, vector<int>, greater<int>> large;  // Min heap

public:
    MedianFinder() {}

    void addNum(int num) {
        small.push(num);

        // Balance: max of small <= min of large
        if (!small.empty() && !large.empty() && small.top() > large.top()) {
            large.push(small.top());
            small.pop();
        }

        // Balance sizes
        if (small.size() > large.size() + 1) {
            large.push(small.top());
            small.pop();
        }
        if (large.size() > small.size()) {
            small.push(large.top());
            large.pop();
        }
    }

    double findMedian() {
        if (small.size() > large.size()) {
            return small.top();
        }
        return (small.top() + large.top()) / 2.0;
    }
};
```

---

<a id="section-14-backtracking"></a>

# 14. BACKTRACKING

---

## Pattern 14A: Subsets

```cpp
vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>> result;
    vector<int> current;
    backtrack(nums, 0, current, result);
    return result;
}

void backtrack(vector<int>& nums, int start, vector<int>& current, vector<vector<int>>& result) {
    result.push_back(current);

    for (int i = start; i < nums.size(); i++) {
        current.push_back(nums[i]);
        backtrack(nums, i + 1, current, result);
        current.pop_back();
    }
}
```

---

## Pattern 14B: Permutations

```cpp
vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> result;
    vector<int> current;
    vector<bool> used(nums.size(), false);
    backtrack(nums, current, used, result);
    return result;
}

void backtrack(vector<int>& nums, vector<int>& current, vector<bool>& used, vector<vector<int>>& result) {
    if (current.size() == nums.size()) {
        result.push_back(current);
        return;
    }

    for (int i = 0; i < nums.size(); i++) {
        if (used[i]) continue;

        used[i] = true;
        current.push_back(nums[i]);
        backtrack(nums, current, used, result);
        current.pop_back();
        used[i] = false;
    }
}
```

---

## Pattern 14C: Combination Sum

```cpp
vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
    vector<vector<int>> result;
    vector<int> current;
    backtrack(candidates, target, 0, current, result);
    return result;
}

void backtrack(vector<int>& candidates, int remaining, int start,
               vector<int>& current, vector<vector<int>>& result) {
    if (remaining == 0) {
        result.push_back(current);
        return;
    }
    if (remaining < 0) return;

    for (int i = start; i < candidates.size(); i++) {
        current.push_back(candidates[i]);
        backtrack(candidates, remaining - candidates[i], i, current, result);
        current.pop_back();
    }
}
```

---

## Pattern 14D: Word Search

```cpp
bool exist(vector<vector<char>>& board, string word) {
    int rows = board.size(), cols = board[0].size();

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (backtrack(board, word, r, c, 0)) {
                return true;
            }
        }
    }

    return false;
}

bool backtrack(vector<vector<char>>& board, string& word, int r, int c, int index) {
    if (index == word.size()) return true;

    if (r < 0 || r >= board.size() || c < 0 || c >= board[0].size() ||
        board[r][c] != word[index]) {
        return false;
    }

    char temp = board[r][c];
    board[r][c] = '#';

    bool found = backtrack(board, word, r + 1, c, index + 1) ||
                 backtrack(board, word, r - 1, c, index + 1) ||
                 backtrack(board, word, r, c + 1, index + 1) ||
                 backtrack(board, word, r, c - 1, index + 1);

    board[r][c] = temp;

    return found;
}
```

---

## Pattern 14E: N-Queens

```cpp
vector<vector<string>> solveNQueens(int n) {
    vector<vector<string>> result;
    vector<string> board(n, string(n, '.'));

    unordered_set<int> cols;
    unordered_set<int> posDiag;
    unordered_set<int> negDiag;

    backtrack(0, n, board, cols, posDiag, negDiag, result);
    return result;
}

void backtrack(int row, int n, vector<string>& board, unordered_set<int>& cols,
               unordered_set<int>& posDiag, unordered_set<int>& negDiag,
               vector<vector<string>>& result) {
    if (row == n) {
        result.push_back(board);
        return;
    }

    for (int col = 0; col < n; col++) {
        if (cols.count(col) || posDiag.count(row + col) || negDiag.count(row - col)) {
            continue;
        }

        cols.insert(col);
        posDiag.insert(row + col);
        negDiag.insert(row - col);
        board[row][col] = 'Q';

        backtrack(row + 1, n, board, cols, posDiag, negDiag, result);

        cols.erase(col);
        posDiag.erase(row + col);
        negDiag.erase(row - col);
        board[row][col] = '.';
    }
}
```

---

<a id="section-15-advanced-graph-patterns"></a>

# 15. ADVANCED GRAPH PATTERNS

| Signal | Algorithm | Complexity |
| ------ | --------- | ---------- |
| Unweighted / many-source shortest path | BFS / multi-source BFS | O(V + E) |
| Edge weights only 0 or 1 | 0-1 BFS | O(V + E) |
| Non-negative weighted shortest path | Dijkstra | O((V + E) log V) |
| Negative edges or at most K edges | Bellman-Ford | O(VE), or O(KE) |
| Repeated merges / connectivity | Union-Find | Almost O(1) per operation |
| Minimum total connection cost | Kruskal / Prim MST | O(E log E) / O(E log V) |
| Critical edges | Tarjan low-link | O(V + E) |
| Use every edge exactly once | Hierholzer | O(E) plus edge ordering |
| Node plus keys/stops/resources | BFS/Dijkstra on full state | Depends on state count |

---

## Pattern 15A: Dijkstra - Non-Negative Weighted Shortest Path

### Problem: Network Delay Time

```cpp
int networkDelayTime(vector<vector<int>>& times, int n, int start) {
    vector<vector<pair<int, int>>> graph(n + 1);
    for (const auto& edge : times) {
        graph[edge[0]].push_back({edge[1], edge[2]});
    }

    const long long INF = LLONG_MAX / 4;
    vector<long long> distance(n + 1, INF);
    distance[start] = 0;

    using State = pair<long long, int>; // distance, node
    priority_queue<State, vector<State>, greater<State>> minHeap;
    minHeap.push({0, start});

    while (!minHeap.empty()) {
        auto [currentDistance, node] = minHeap.top();
        minHeap.pop();
        if (currentDistance != distance[node]) continue; // stale entry

        for (auto [neighbour, weight] : graph[node]) {
            long long candidate = currentDistance + weight;
            if (candidate < distance[neighbour]) {
                distance[neighbour] = candidate;
                minHeap.push({candidate, neighbour});
            }
        }
    }

    long long answer = *max_element(distance.begin() + 1, distance.end());
    return answer == INF ? -1 : static_cast<int>(answer);
}
```

**Invariant:** the current best heap removal is final with non-negative weights. Never mark a node visited merely when it enters the heap.

---

## Pattern 15B: 0-1 BFS - Binary Edge Weights

### Problem: Minimum Obstacle Removal to Reach Corner

```cpp
int minimumObstacles(vector<vector<int>>& grid) {
    int rows = grid.size(), cols = grid[0].size();
    vector<vector<int>> distance(rows, vector<int>(cols, INT_MAX));
    deque<pair<int, int>> worklist;
    distance[0][0] = 0;
    worklist.push_front({0, 0});
    int directions[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    while (!worklist.empty()) {
        auto [row, col] = worklist.front();
        worklist.pop_front();

        for (auto& direction : directions) {
            int nextRow = row + direction[0], nextCol = col + direction[1];
            if (nextRow < 0 || nextRow >= rows || nextCol < 0 || nextCol >= cols) continue;

            int weight = grid[nextRow][nextCol];
            int candidate = distance[row][col] + weight;
            if (candidate >= distance[nextRow][nextCol]) continue;

            distance[nextRow][nextCol] = candidate;
            if (weight == 0) worklist.push_front({nextRow, nextCol});
            else worklist.push_back({nextRow, nextCol});
        }
    }
    return distance[rows - 1][cols - 1];
}
```

Zero-cost edges go to the front and cost-one edges to the back, maintaining Dijkstra's order in O(V + E).

---

## Pattern 15C: Bellman-Ford by Layers - Bounded Stops

### Problem: Cheapest Flights Within K Stops

```cpp
int findCheapestPrice(
    int n, vector<vector<int>>& flights, int source, int target, int k
) {
    const long long INF = LLONG_MAX / 4;
    vector<long long> cost(n, INF);
    cost[source] = 0;

    for (int edgesUsed = 0; edgesUsed <= k; ++edgesUsed) {
        vector<long long> nextCost = cost;
        for (const auto& flight : flights) {
            int from = flight[0], to = flight[1], price = flight[2];
            if (cost[from] != INF) {
                nextCost[to] = min(nextCost[to], cost[from] + price);
            }
        }
        cost.swap(nextCost);
    }
    return cost[target] == INF ? -1 : static_cast<int>(cost[target]);
}
```

**Critical detail:** read from the old layer and write to the new one. In-place relaxation may use too many edges in one round.

---

## Pattern 15D: Union-Find - Dynamic Connectivity

### Problem: Redundant Connection

```cpp
class UnionFind {
    vector<int> parent;
    vector<int> componentSize;

public:
    explicit UnionFind(int n) : parent(n), componentSize(n, 1) {
        iota(parent.begin(), parent.end(), 0);
    }

    int find(int node) {
        if (parent[node] != node) parent[node] = find(parent[node]);
        return parent[node];
    }

    bool unite(int first, int second) {
        int rootFirst = find(first), rootSecond = find(second);
        if (rootFirst == rootSecond) return false;
        if (componentSize[rootFirst] < componentSize[rootSecond]) swap(rootFirst, rootSecond);
        parent[rootSecond] = rootFirst;
        componentSize[rootFirst] += componentSize[rootSecond];
        return true;
    }
};

vector<int> findRedundantConnection(vector<vector<int>>& edges) {
    UnionFind unionFind(edges.size() + 1);
    for (const auto& edge : edges) {
        if (!unionFind.unite(edge[0], edge[1])) return edge;
    }
    return {};
}
```

An edge whose endpoints already share a root closes an undirected cycle. Path compression and union by size are near constant amortised time.

---

## Pattern 15E: Prim's Minimum Spanning Tree

### Problem: Min Cost to Connect All Points

```cpp
int minCostConnectPoints(vector<vector<int>>& points) {
    int n = points.size();
    vector<bool> inTree(n, false);
    using Edge = pair<int, int>; // cost, destination
    priority_queue<Edge, vector<Edge>, greater<Edge>> minHeap;
    minHeap.push({0, 0});

    int totalCost = 0, used = 0;
    while (used < n) {
        auto [cost, point] = minHeap.top();
        minHeap.pop();
        if (inTree[point]) continue;

        inTree[point] = true;
        totalCost += cost;
        ++used;

        for (int neighbour = 0; neighbour < n; ++neighbour) {
            if (!inTree[neighbour]) {
                int distance = abs(points[point][0] - points[neighbour][0])
                    + abs(points[point][1] - points[neighbour][1]);
                minHeap.push({distance, neighbour});
            }
        }
    }
    return totalCost;
}
```

MST minimises total network cost; Dijkstra minimises distances from one source. They are not interchangeable.

---

## Pattern 15F: Bipartite Graph - Two-Colouring

```cpp
bool isBipartite(vector<vector<int>>& graph) {
    vector<int> colour(graph.size(), -1);

    for (int start = 0; start < static_cast<int>(graph.size()); ++start) {
        if (colour[start] != -1) continue;
        queue<int> nodes;
        nodes.push(start);
        colour[start] = 0;

        while (!nodes.empty()) {
            int node = nodes.front();
            nodes.pop();
            for (int neighbour : graph[node]) {
                if (colour[neighbour] == -1) {
                    colour[neighbour] = colour[node] ^ 1;
                    nodes.push(neighbour);
                } else if (colour[neighbour] == colour[node]) {
                    return false;
                }
            }
        }
    }
    return true;
}
```

Every edge must cross colours. Looping over all starts handles disconnected graphs; a conflict is an odd cycle.

---

## Pattern 15G: Tarjan Low-Link - Bridges

```cpp
class BridgeFinder {
    vector<vector<pair<int, int>>> graph; // neighbour, edge ID
    vector<int> discovery;
    vector<int> low;
    vector<vector<int>> bridges;
    int timer = 0;

    void dfs(int node, int parentEdge) {
        discovery[node] = low[node] = timer++;
        for (auto [neighbour, edgeId] : graph[node]) {
            if (edgeId == parentEdge) continue;
            if (discovery[neighbour] == -1) {
                dfs(neighbour, edgeId);
                low[node] = min(low[node], low[neighbour]);
                if (low[neighbour] > discovery[node]) bridges.push_back({node, neighbour});
            } else {
                low[node] = min(low[node], discovery[neighbour]);
            }
        }
    }

public:
    vector<vector<int>> criticalConnections(int n, vector<vector<int>>& connections) {
        bridges.clear();
        timer = 0;
        graph.assign(n, {});
        discovery.assign(n, -1);
        low.resize(n);
        for (int edgeId = 0; edgeId < static_cast<int>(connections.size()); ++edgeId) {
            int first = connections[edgeId][0], second = connections[edgeId][1];
            graph[first].push_back({second, edgeId});
            graph[second].push_back({first, edgeId});
        }
        for (int node = 0; node < n; ++node) {
            if (discovery[node] == -1) dfs(node, -1);
        }
        return bridges;
    }
};
```

`low[v] > discovery[u]` means the DFS subtree at `v` cannot bypass edge `u-v`. Parent edge IDs make the template safe for parallel edges.

---

## Pattern 15H: State-Space BFS - Node Plus Bitmask

### Problem: Shortest Path to Get All Keys

```cpp
int shortestPathAllKeys(vector<string>& grid) {
    int rows = grid.size(), cols = grid[0].size();
    int startRow = 0, startCol = 0, allKeys = 0;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            char cell = grid[row][col];
            if (cell == '@') startRow = row, startCol = col;
            else if (cell >= 'a' && cell <= 'f') allKeys |= 1 << (cell - 'a');
        }
    }

    queue<tuple<int, int, int>> worklist;
    worklist.push({startRow, startCol, 0});
    vector visited(rows, vector(cols, vector<bool>(64, false)));
    visited[startRow][startCol][0] = true;
    int directions[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    int steps = 0;

    while (!worklist.empty()) {
        int levelSize = worklist.size();
        while (levelSize--) {
            auto [row, col, keys] = worklist.front();
            worklist.pop();
            if (keys == allKeys) return steps;

            for (auto& direction : directions) {
                int nextRow = row + direction[0], nextCol = col + direction[1];
                if (nextRow < 0 || nextRow >= rows || nextCol < 0 || nextCol >= cols) continue;
                char cell = grid[nextRow][nextCol];
                if (cell == '#') continue;
                int nextKeys = keys;
                if (cell >= 'a' && cell <= 'f') nextKeys |= 1 << (cell - 'a');
                if (cell >= 'A' && cell <= 'F' && !(keys & (1 << (cell - 'A')))) continue;
                if (!visited[nextRow][nextCol][nextKeys]) {
                    visited[nextRow][nextCol][nextKeys] = true;
                    worklist.push({nextRow, nextCol, nextKeys});
                }
            }
        }
        ++steps;
    }
    return -1;
}
```

The real vertex is `(row, col, keys)`, not just `(row, col)`. The full state determines future legal moves.

---

## Pattern 15I: Eulerian Path - Use Every Edge Once

### Problem: Reconstruct Itinerary

```cpp
vector<string> findItinerary(vector<vector<string>>& tickets) {
    unordered_map<string, priority_queue<string, vector<string>, greater<string>>> graph;
    for (const auto& ticket : tickets) graph[ticket[0]].push(ticket[1]);
    vector<string> route;

    function<void(const string&)> visit = [&](const string& airport) {
        auto& destinations = graph[airport];
        while (!destinations.empty()) {
            string next = destinations.top();
            destinations.pop();
            visit(next);
        }
        route.push_back(airport);
    };

    visit("JFK");
    reverse(route.begin(), route.end());
    return route;
}
```

Hierholzer appends a node after consuming its outgoing edges, so the route is built backwards. Eulerian means every **edge** once; Hamiltonian means every **vertex** once.

---

## Advanced Graph Interview Checklist

1. Define the real vertex and edge, including hidden state.
2. Match edge weights to BFS, 0-1 BFS, Dijkstra, Bellman-Ford, or DAG relaxation.
3. Cover every disconnected component where required.
4. Skip stale Dijkstra entries and mark ordinary BFS states on enqueue.
5. Distinguish MST total cost from single-source shortest paths.
6. Give complexity in V and E—or in the expanded number of states.

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

<a id="section-c-plus-plus-syntax-cheat-sheet"></a>

# C++ Syntax Cheat Sheet

```cpp
// unordered_map
unordered_map<K, V> map;
map[key] = value;
map[key];  // access
map.count(key);  // check existence
map.erase(key);
for (auto& [k, v] : map) { }  // iterate

// unordered_set
unordered_set<T> set;
set.insert(value);
set.count(value);
set.erase(value);

// vector
vector<T> vec;
vec.push_back(value);
vec[index];
vec.back();
vec.pop_back();
vec.size();
vec.empty();

// queue
queue<T> q;
q.push(value);
q.front();
q.pop();
q.empty();

// deque
deque<T> dq;
dq.push_front(value);
dq.push_back(value);
dq.pop_front();
dq.pop_back();
dq.front();
dq.back();

// stack
stack<T> stk;
stk.push(value);
stk.top();
stk.pop();
stk.empty();

// priority_queue
priority_queue<T> maxHeap;  // max heap by default
priority_queue<T, vector<T>, greater<T>> minHeap;
priority_queue<pair<int,int>, vector<pair<int,int>>, greater<pair<int,int>>> pq;
pq.push(value);
pq.top();
pq.pop();

// Algorithms
sort(vec.begin(), vec.end());
fill(vec.begin(), vec.end(), value);
*max_element(vec.begin(), vec.end());
*min_element(vec.begin(), vec.end());
reverse(vec.begin(), vec.end());

// String
s[i];  // char at index
s.substr(start, length);
to_string(num);
stoi(s);
s.size();
s.empty();

// Common includes
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <stack>
#include <algorithm>
#include <climits>
```
