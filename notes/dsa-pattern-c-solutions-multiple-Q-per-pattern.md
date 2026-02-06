# Complete DSA Patterns: Multiple Problems Per Pattern (C)

**Note:** C does not have built-in data structures like hash maps, dynamic arrays, or queues. This guide uses arrays and structs with manual memory management. For production code, consider implementing or using libraries for these data structures.

---

# 1. ARRAYS & HASHING

---

## Pattern 1A: Complement Lookup (Two Sum Pattern)

### Problem: Two Sum

**Statement:** Given array `nums` and `target`, return indices of two numbers that add up to target.

```
Input: nums = [2, 7, 11, 15], target = 9
Output: [0, 1]
```

```c
// Simple O(n^2) solution without hash map
int* twoSum(int* nums, int numsSize, int target, int* returnSize) {
    int* result = (int*)malloc(2 * sizeof(int));
    *returnSize = 2;

    for (int i = 0; i < numsSize; i++) {
        for (int j = i + 1; j < numsSize; j++) {
            if (nums[i] + nums[j] == target) {
                result[0] = i;
                result[1] = j;
                return result;
            }
        }
    }

    *returnSize = 0;
    free(result);
    return NULL;
}

// O(n) solution with simple hash table (for positive numbers in limited range)
#define HASH_SIZE 10007

typedef struct HashNode {
    int key;
    int value;
    struct HashNode* next;
} HashNode;

int hash(int key) {
    return ((key % HASH_SIZE) + HASH_SIZE) % HASH_SIZE;
}

int* twoSumOptimized(int* nums, int numsSize, int target, int* returnSize) {
    HashNode* hashTable[HASH_SIZE] = {NULL};
    int* result = (int*)malloc(2 * sizeof(int));
    *returnSize = 2;

    for (int i = 0; i < numsSize; i++) {
        int complement = target - nums[i];
        int h = hash(complement);

        // Search for complement
        HashNode* node = hashTable[h];
        while (node != NULL) {
            if (node->key == complement) {
                result[0] = node->value;
                result[1] = i;
                // Free hash table and return
                return result;
            }
            node = node->next;
        }

        // Insert current number
        h = hash(nums[i]);
        HashNode* newNode = (HashNode*)malloc(sizeof(HashNode));
        newNode->key = nums[i];
        newNode->value = i;
        newNode->next = hashTable[h];
        hashTable[h] = newNode;
    }

    *returnSize = 0;
    free(result);
    return NULL;
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

```c
typedef struct {
    int num;
    int freq;
} NumFreq;

int compareFreq(const void* a, const void* b) {
    return ((NumFreq*)b)->freq - ((NumFreq*)a)->freq;
}

int* topKFrequent(int* nums, int numsSize, int k, int* returnSize) {
    // Count frequencies using sorting approach
    int* sorted = (int*)malloc(numsSize * sizeof(int));
    memcpy(sorted, nums, numsSize * sizeof(int));
    qsort(sorted, numsSize, sizeof(int), compareInt);

    // Count unique elements
    NumFreq* freqs = (NumFreq*)malloc(numsSize * sizeof(NumFreq));
    int uniqueCount = 0;
    int i = 0;

    while (i < numsSize) {
        freqs[uniqueCount].num = sorted[i];
        freqs[uniqueCount].freq = 1;
        while (i + 1 < numsSize && sorted[i + 1] == sorted[i]) {
            freqs[uniqueCount].freq++;
            i++;
        }
        uniqueCount++;
        i++;
    }

    // Sort by frequency
    qsort(freqs, uniqueCount, sizeof(NumFreq), compareFreq);

    // Get top k
    int* result = (int*)malloc(k * sizeof(int));
    for (int j = 0; j < k; j++) {
        result[j] = freqs[j].num;
    }

    *returnSize = k;
    free(sorted);
    free(freqs);
    return result;
}

int compareInt(const void* a, const void* b) {
    return *(int*)a - *(int*)b;
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

```c
// Helper to create sorted key
void getSortedKey(const char* str, char* key) {
    strcpy(key, str);
    int len = strlen(key);
    // Simple bubble sort for small strings
    for (int i = 0; i < len - 1; i++) {
        for (int j = 0; j < len - i - 1; j++) {
            if (key[j] > key[j + 1]) {
                char temp = key[j];
                key[j] = key[j + 1];
                key[j + 1] = temp;
            }
        }
    }
}

// Note: Full implementation requires dynamic arrays and hash maps
// This is a simplified version showing the concept
char*** groupAnagrams(char** strs, int strsSize, int* returnSize, int** returnColumnSizes) {
    if (strsSize == 0) {
        *returnSize = 0;
        return NULL;
    }

    // Create array of (sorted_key, original_index) pairs
    typedef struct {
        char key[101];
        int index;
    } KeyIndex;

    KeyIndex* pairs = (KeyIndex*)malloc(strsSize * sizeof(KeyIndex));
    for (int i = 0; i < strsSize; i++) {
        getSortedKey(strs[i], pairs[i].key);
        pairs[i].index = i;
    }

    // Sort by key
    qsort(pairs, strsSize, sizeof(KeyIndex), 
          (int (*)(const void*, const void*))strcmp);

    // Count groups
    int groupCount = 1;
    for (int i = 1; i < strsSize; i++) {
        if (strcmp(pairs[i].key, pairs[i-1].key) != 0) {
            groupCount++;
        }
    }

    // Allocate result
    char*** result = (char***)malloc(groupCount * sizeof(char**));
    *returnColumnSizes = (int*)malloc(groupCount * sizeof(int));
    *returnSize = groupCount;

    // Fill groups (simplified)
    // ... implementation continues ...

    free(pairs);
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

```c
#define HASH_SIZE 10007

typedef struct HashNode {
    int key;
    int count;
    struct HashNode* next;
} HashNode;

int hashFunc(int key) {
    return ((key % HASH_SIZE) + HASH_SIZE) % HASH_SIZE;
}

int subarraySum(int* nums, int numsSize, int k) {
    HashNode* hashTable[HASH_SIZE] = {NULL};
    int count = 0;
    int prefixSum = 0;

    // Insert initial state (sum 0 seen once)
    HashNode* initial = (HashNode*)malloc(sizeof(HashNode));
    initial->key = 0;
    initial->count = 1;
    initial->next = NULL;
    hashTable[hashFunc(0)] = initial;

    for (int i = 0; i < numsSize; i++) {
        prefixSum += nums[i];

        // Look for (prefixSum - k)
        int target = prefixSum - k;
        int h = hashFunc(target);
        HashNode* node = hashTable[h];
        while (node != NULL) {
            if (node->key == target) {
                count += node->count;
                break;
            }
            node = node->next;
        }

        // Insert/update prefixSum
        h = hashFunc(prefixSum);
        node = hashTable[h];
        while (node != NULL) {
            if (node->key == prefixSum) {
                node->count++;
                break;
            }
            node = node->next;
        }
        if (node == NULL) {
            HashNode* newNode = (HashNode*)malloc(sizeof(HashNode));
            newNode->key = prefixSum;
            newNode->count = 1;
            newNode->next = hashTable[h];
            hashTable[h] = newNode;
        }
    }

    // Free hash table (in production code)
    return count;
}
```

**Keywords:** "subarray sum equals", "contiguous elements sum to"
**Intuition:** prefix[j] - prefix[i] = sum of subarray [i+1, j]. Store prefix sums; look for prefixSum - k.

---

# 2. TWO POINTERS

---

## Pattern 2A: Opposite Ends (Sorted or Optimization)

### Problem: Container With Most Water

**Statement:** Given `height[i]`, find two lines that form container holding most water.

```c
int maxArea(int* height, int heightSize) {
    int left = 0, right = heightSize - 1;
    int maxWater = 0;

    while (left < right) {
        int width = right - left;
        int h = height[left] < height[right] ? height[left] : height[right];
        int area = width * h;
        if (area > maxWater) maxWater = area;

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

```c
int compare(const void* a, const void* b) {
    return *(int*)a - *(int*)b;
}

int** threeSum(int* nums, int numsSize, int* returnSize, int** returnColumnSizes) {
    qsort(nums, numsSize, sizeof(int), compare);

    // Allocate maximum possible results
    int capacity = numsSize * numsSize;
    int** result = (int**)malloc(capacity * sizeof(int*));
    *returnColumnSizes = (int*)malloc(capacity * sizeof(int));
    *returnSize = 0;

    for (int i = 0; i < numsSize - 2; i++) {
        // Skip duplicates
        if (i > 0 && nums[i] == nums[i - 1]) continue;

        int left = i + 1, right = numsSize - 1;
        int target = -nums[i];

        while (left < right) {
            int sum = nums[left] + nums[right];

            if (sum == target) {
                result[*returnSize] = (int*)malloc(3 * sizeof(int));
                result[*returnSize][0] = nums[i];
                result[*returnSize][1] = nums[left];
                result[*returnSize][2] = nums[right];
                (*returnColumnSizes)[*returnSize] = 3;
                (*returnSize)++;

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

```c
int removeDuplicates(int* nums, int numsSize) {
    if (numsSize == 0) return 0;

    int slow = 0;

    for (int fast = 1; fast < numsSize; fast++) {
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

```c
void moveZeroes(int* nums, int numsSize) {
    int slow = 0;

    for (int fast = 0; fast < numsSize; fast++) {
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

# 3. SLIDING WINDOW

---

## Pattern 3A: Fixed Size Window

### Problem: Maximum Sum Subarray of Size K

**Statement:** Find maximum sum of any contiguous subarray of size `k`.

```c
int maxSumSubarray(int* nums, int numsSize, int k) {
    if (numsSize < k) return 0;

    // Calculate first window
    int windowSum = 0;
    for (int i = 0; i < k; i++) {
        windowSum += nums[i];
    }
    int maxSum = windowSum;

    // Slide the window
    for (int i = k; i < numsSize; i++) {
        windowSum += nums[i] - nums[i - k];
        if (windowSum > maxSum) maxSum = windowSum;
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

```c
int lengthOfLongestSubstring(char* s) {
    int charIndex[128];  // ASCII characters
    memset(charIndex, -1, sizeof(charIndex));

    int left = 0;
    int maxLength = 0;
    int len = strlen(s);

    for (int right = 0; right < len; right++) {
        char c = s[right];

        if (charIndex[(int)c] >= left) {
            left = charIndex[(int)c] + 1;
        }

        charIndex[(int)c] = right;
        int length = right - left + 1;
        if (length > maxLength) maxLength = length;
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

```c
int characterReplacement(char* s, int k) {
    int count[26] = {0};
    int left = 0;
    int maxFreq = 0;
    int maxLength = 0;
    int len = strlen(s);

    for (int right = 0; right < len; right++) {
        count[s[right] - 'A']++;
        if (count[s[right] - 'A'] > maxFreq) {
            maxFreq = count[s[right] - 'A'];
        }

        // Window size - maxFreq = chars to replace
        while ((right - left + 1) - maxFreq > k) {
            count[s[left] - 'A']--;
            left++;
        }

        int length = right - left + 1;
        if (length > maxLength) maxLength = length;
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

```c
char* minWindow(char* s, char* t) {
    if (strlen(s) == 0 || strlen(t) == 0) return "";

    int need[128] = {0};
    int have[128] = {0};
    int tLen = strlen(t);
    int sLen = strlen(s);

    int required = 0;
    for (int i = 0; i < tLen; i++) {
        if (need[(int)t[i]] == 0) required++;
        need[(int)t[i]]++;
    }

    int formed = 0;
    int left = 0;
    int minLen = INT_MAX;
    int resultLeft = 0;

    for (int right = 0; right < sLen; right++) {
        char c = s[right];
        have[(int)c]++;

        if (need[(int)c] > 0 && have[(int)c] == need[(int)c]) {
            formed++;
        }

        while (formed == required) {
            if (right - left + 1 < minLen) {
                minLen = right - left + 1;
                resultLeft = left;
            }

            char leftChar = s[left];
            have[(int)leftChar]--;
            if (need[(int)leftChar] > 0 && have[(int)leftChar] < need[(int)leftChar]) {
                formed--;
            }
            left++;
        }
    }

    if (minLen == INT_MAX) return "";

    char* result = (char*)malloc((minLen + 1) * sizeof(char));
    strncpy(result, s + resultLeft, minLen);
    result[minLen] = '\0';
    return result;
}
```

**Keywords:** "minimum window", "shortest substring containing", "smallest subarray with"
**Intuition:** Expand until valid. Then shrink while still valid, tracking minimum.

---

# 4. BINARY SEARCH

---

## Pattern 4A: Standard Binary Search

### Problem: Binary Search

**Statement:** Find target in sorted array. Return index or -1.

```c
int search(int* nums, int numsSize, int target) {
    int left = 0, right = numsSize - 1;

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

```c
int findLeft(int* nums, int numsSize, int target) {
    int left = 0, right = numsSize - 1;
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

int findRight(int* nums, int numsSize, int target) {
    int left = 0, right = numsSize - 1;
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

int* searchRange(int* nums, int numsSize, int target, int* returnSize) {
    int* result = (int*)malloc(2 * sizeof(int));
    *returnSize = 2;
    result[0] = findLeft(nums, numsSize, target);
    result[1] = findRight(nums, numsSize, target);
    return result;
}
```

**Keywords:** "first occurrence", "last occurrence", "leftmost/rightmost"
**Intuition:** Modify binary search to keep going even after finding target.

---

## Pattern 4C: Search in Rotated Sorted Array

### Problem: Search in Rotated Sorted Array

**Statement:** Array was sorted then rotated. Find target index.

```c
int searchRotated(int* nums, int numsSize, int target) {
    int left = 0, right = numsSize - 1;

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

```c
int canFinish(int* piles, int pilesSize, int speed, int h) {
    long long hours = 0;
    for (int i = 0; i < pilesSize; i++) {
        hours += (piles[i] + speed - 1) / speed;  // Ceiling division
    }
    return hours <= h;
}

int minEatingSpeed(int* piles, int pilesSize, int h) {
    int left = 1;
    int right = 0;
    for (int i = 0; i < pilesSize; i++) {
        if (piles[i] > right) right = piles[i];
    }

    while (left < right) {
        int mid = left + (right - left) / 2;

        if (canFinish(piles, pilesSize, mid, h)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }

    return left;
}
```

**Keywords:** "minimum speed/capacity that satisfies", "minimum X such that"
**Intuition:** Binary search on the answer space, not the input. Check if answer works.

---

# 5. STACKS

---

## Pattern 5A: Matching Pairs (Parentheses)

### Problem: Valid Parentheses

**Statement:** Determine if string of brackets is valid.

```c
bool isValid(char* s) {
    int len = strlen(s);
    char* stack = (char*)malloc(len * sizeof(char));
    int top = -1;

    for (int i = 0; i < len; i++) {
        char c = s[i];

        if (c == '(' || c == '[' || c == '{') {
            stack[++top] = c;
        } else {
            if (top == -1) {
                free(stack);
                return false;
            }

            char expected;
            if (c == ')') expected = '(';
            else if (c == ']') expected = '[';
            else expected = '{';

            if (stack[top] != expected) {
                free(stack);
                return false;
            }
            top--;
        }
    }

    bool result = (top == -1);
    free(stack);
    return result;
}
```

**Keywords:** "valid parentheses", "balanced brackets", "matching"
**Intuition:** Push opening brackets. Pop and match on closing brackets.

---

## Pattern 5B: Monotonic Stack - Next Greater Element

### Problem: Daily Temperatures

**Statement:** For each day, find days until warmer temperature.

```c
int* dailyTemperatures(int* temperatures, int temperaturesSize, int* returnSize) {
    int* answer = (int*)calloc(temperaturesSize, sizeof(int));
    int* stack = (int*)malloc(temperaturesSize * sizeof(int));  // Indices
    int top = -1;
    *returnSize = temperaturesSize;

    for (int i = 0; i < temperaturesSize; i++) {
        while (top >= 0 && temperatures[i] > temperatures[stack[top]]) {
            int prevDay = stack[top--];
            answer[prevDay] = i - prevDay;
        }
        stack[++top] = i;
    }

    free(stack);
    return answer;
}
```

**Keywords:** "next greater", "next warmer", "days until"
**Intuition:** Stack holds unresolved elements. Pop when current element resolves them.

---

## Pattern 5C: Monotonic Stack - Largest Rectangle

### Problem: Largest Rectangle in Histogram

**Statement:** Find largest rectangle area in histogram.

```c
int largestRectangleArea(int* heights, int heightsSize) {
    int* stack = (int*)malloc((heightsSize + 1) * sizeof(int));
    int top = -1;
    int maxArea = 0;

    for (int i = 0; i <= heightsSize; i++) {
        int h = (i == heightsSize) ? 0 : heights[i];

        while (top >= 0 && h < heights[stack[top]]) {
            int height = heights[stack[top--]];
            int width = (top == -1) ? i : i - stack[top] - 1;
            int area = height * width;
            if (area > maxArea) maxArea = area;
        }

        stack[++top] = i;
    }

    free(stack);
    return maxArea;
}
```

**Keywords:** "largest rectangle", "histogram", "maximum area"
**Intuition:** For each bar, find how far left and right it can extend. Monotonic stack finds boundaries.

---

## Pattern 5D: Min Stack (Design with O(1) Operations)

### Problem: Min Stack

**Statement:** Design stack supporting push, pop, top, and getMin in O(1).

```c
typedef struct {
    int* data;
    int* minData;
    int top;
    int capacity;
} MinStack;

MinStack* minStackCreate() {
    MinStack* stack = (MinStack*)malloc(sizeof(MinStack));
    stack->capacity = 1000;
    stack->data = (int*)malloc(stack->capacity * sizeof(int));
    stack->minData = (int*)malloc(stack->capacity * sizeof(int));
    stack->top = -1;
    return stack;
}

void minStackPush(MinStack* obj, int val) {
    obj->top++;
    obj->data[obj->top] = val;

    if (obj->top == 0 || val <= obj->minData[obj->top - 1]) {
        obj->minData[obj->top] = val;
    } else {
        obj->minData[obj->top] = obj->minData[obj->top - 1];
    }
}

void minStackPop(MinStack* obj) {
    obj->top--;
}

int minStackTop(MinStack* obj) {
    return obj->data[obj->top];
}

int minStackGetMin(MinStack* obj) {
    return obj->minData[obj->top];
}

void minStackFree(MinStack* obj) {
    free(obj->data);
    free(obj->minData);
    free(obj);
}
```

**Keywords:** "min in O(1)", "design stack", "constant time minimum"
**Intuition:** Maintain parallel stack tracking minimum at each state.

---

# 6. QUEUES

---

## Pattern 6A: BFS with Queue

### Problem: Rotting Oranges

**Statement:** Grid with 0=empty, 1=fresh, 2=rotten. Rot spreads each minute. Time until all rotten?

```c
int orangesRotting(int** grid, int gridSize, int* gridColSize) {
    int rows = gridSize, cols = gridColSize[0];

    // Queue using array
    int* queueR = (int*)malloc(rows * cols * sizeof(int));
    int* queueC = (int*)malloc(rows * cols * sizeof(int));
    int front = 0, back = 0;
    int fresh = 0;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (grid[r][c] == 2) {
                queueR[back] = r;
                queueC[back] = c;
                back++;
            } else if (grid[r][c] == 1) {
                fresh++;
            }
        }
    }

    if (fresh == 0) {
        free(queueR);
        free(queueC);
        return 0;
    }

    int minutes = 0;
    int dr[] = {-1, 0, 1, 0};
    int dc[] = {0, 1, 0, -1};

    while (front < back) {
        int size = back - front;
        int rotted = 0;

        for (int i = 0; i < size; i++) {
            int r = queueR[front];
            int c = queueC[front];
            front++;

            for (int d = 0; d < 4; d++) {
                int nr = r + dr[d];
                int nc = c + dc[d];

                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 1) {
                    grid[nr][nc] = 2;
                    queueR[back] = nr;
                    queueC[back] = nc;
                    back++;
                    fresh--;
                    rotted = 1;
                }
            }
        }

        if (rotted) minutes++;
    }

    free(queueR);
    free(queueC);
    return fresh == 0 ? minutes : -1;
}
```

**Keywords:** "spreads", "minimum time", "level by level", "simultaneous"
**Intuition:** Multi-source BFS. All sources start together. Each level = 1 time unit.

---

## Pattern 6B: Monotonic Deque - Sliding Window Maximum

### Problem: Sliding Window Maximum

**Statement:** Return max of each sliding window of size k.

```c
int* maxSlidingWindow(int* nums, int numsSize, int k, int* returnSize) {
    *returnSize = numsSize - k + 1;
    int* result = (int*)malloc(*returnSize * sizeof(int));
    int* deque = (int*)malloc(numsSize * sizeof(int));  // Indices
    int front = 0, back = 0;

    for (int i = 0; i < numsSize; i++) {
        // Remove elements smaller than current
        while (back > front && nums[deque[back - 1]] < nums[i]) {
            back--;
        }

        deque[back++] = i;

        // Remove elements outside window
        if (deque[front] <= i - k) {
            front++;
        }

        // Record result
        if (i >= k - 1) {
            result[i - k + 1] = nums[deque[front]];
        }
    }

    free(deque);
    return result;
}
```

**Keywords:** "sliding window maximum/minimum", "max of each window"
**Intuition:** Monotonic deque keeps potential maxes in decreasing order. Front is always max.

---

# 7. LINKED LISTS

---

## Pattern 7A: Reverse Linked List

```c
struct ListNode* reverseList(struct ListNode* head) {
    struct ListNode* prev = NULL;
    struct ListNode* curr = head;

    while (curr != NULL) {
        struct ListNode* next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }

    return prev;
}
```

---

## Pattern 7B: Fast/Slow - Cycle Detection

```c
bool hasCycle(struct ListNode* head) {
    struct ListNode* slow = head;
    struct ListNode* fast = head;

    while (fast != NULL && fast->next != NULL) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return true;
    }

    return false;
}
```

---

## Pattern 7C: Fast/Slow - Find Middle

```c
struct ListNode* middleNode(struct ListNode* head) {
    struct ListNode* slow = head;
    struct ListNode* fast = head;

    while (fast != NULL && fast->next != NULL) {
        slow = slow->next;
        fast = fast->next->next;
    }

    return slow;
}
```

---

## Pattern 7D: Two Pointers with Gap

### Problem: Remove Nth Node From End

```c
struct ListNode* removeNthFromEnd(struct ListNode* head, int n) {
    struct ListNode dummy;
    dummy.val = 0;
    dummy.next = head;

    struct ListNode* first = &dummy;
    struct ListNode* second = &dummy;

    for (int i = 0; i <= n; i++) {
        first = first->next;
    }

    while (first != NULL) {
        first = first->next;
        second = second->next;
    }

    struct ListNode* toDelete = second->next;
    second->next = second->next->next;
    free(toDelete);

    return dummy.next;
}
```

---

## Pattern 7E: Merge Two Sorted Lists

```c
struct ListNode* mergeTwoLists(struct ListNode* list1, struct ListNode* list2) {
    struct ListNode dummy;
    dummy.next = NULL;
    struct ListNode* curr = &dummy;

    while (list1 != NULL && list2 != NULL) {
        if (list1->val <= list2->val) {
            curr->next = list1;
            list1 = list1->next;
        } else {
            curr->next = list2;
            list2 = list2->next;
        }
        curr = curr->next;
    }

    curr->next = (list1 != NULL) ? list1 : list2;

    return dummy.next;
}
```

---

# 8. TREES - DFS

---

## Pattern 8A: Basic DFS (Max Depth)

```c
int maxDepth(struct TreeNode* root) {
    if (root == NULL) return 0;
    int leftDepth = maxDepth(root->left);
    int rightDepth = maxDepth(root->right);
    return 1 + (leftDepth > rightDepth ? leftDepth : rightDepth);
}
```

---

## Pattern 8B: DFS with Validation (BST)

```c
bool validate(struct TreeNode* node, long minVal, long maxVal) {
    if (node == NULL) return true;

    if (node->val <= minVal || node->val >= maxVal) return false;

    return validate(node->left, minVal, node->val) &&
           validate(node->right, node->val, maxVal);
}

bool isValidBST(struct TreeNode* root) {
    return validate(root, LONG_MIN, LONG_MAX);
}
```

---

## Pattern 8C: DFS - Lowest Common Ancestor

```c
struct TreeNode* lowestCommonAncestor(struct TreeNode* root, struct TreeNode* p, struct TreeNode* q) {
    if (root == NULL || root == p || root == q) return root;

    struct TreeNode* left = lowestCommonAncestor(root->left, p, q);
    struct TreeNode* right = lowestCommonAncestor(root->right, p, q);

    if (left != NULL && right != NULL) return root;

    return left != NULL ? left : right;
}
```

---

## Pattern 8D: DFS - Path Sum

```c
void dfs(struct TreeNode* node, int remaining, int* path, int pathLen,
         int** result, int* returnSize, int** returnColumnSizes) {
    if (node == NULL) return;

    path[pathLen++] = node->val;

    if (node->left == NULL && node->right == NULL && remaining == node->val) {
        result[*returnSize] = (int*)malloc(pathLen * sizeof(int));
        memcpy(result[*returnSize], path, pathLen * sizeof(int));
        (*returnColumnSizes)[*returnSize] = pathLen;
        (*returnSize)++;
    }

    dfs(node->left, remaining - node->val, path, pathLen, result, returnSize, returnColumnSizes);
    dfs(node->right, remaining - node->val, path, pathLen, result, returnSize, returnColumnSizes);
}

int** pathSum(struct TreeNode* root, int targetSum, int* returnSize, int** returnColumnSizes) {
    int** result = (int**)malloc(1000 * sizeof(int*));
    *returnColumnSizes = (int*)malloc(1000 * sizeof(int));
    *returnSize = 0;
    int* path = (int*)malloc(1000 * sizeof(int));

    dfs(root, targetSum, path, 0, result, returnSize, returnColumnSizes);

    free(path);
    return result;
}
```

---

# 9. TREES - BFS

---

## Pattern 9A: Level Order Traversal

```c
int** levelOrder(struct TreeNode* root, int* returnSize, int** returnColumnSizes) {
    if (root == NULL) {
        *returnSize = 0;
        return NULL;
    }

    int** result = (int**)malloc(2000 * sizeof(int*));
    *returnColumnSizes = (int*)malloc(2000 * sizeof(int));
    *returnSize = 0;

    struct TreeNode** queue = (struct TreeNode**)malloc(10000 * sizeof(struct TreeNode*));
    int front = 0, back = 0;
    queue[back++] = root;

    while (front < back) {
        int size = back - front;
        result[*returnSize] = (int*)malloc(size * sizeof(int));
        (*returnColumnSizes)[*returnSize] = size;

        for (int i = 0; i < size; i++) {
            struct TreeNode* node = queue[front++];
            result[*returnSize][i] = node->val;
            if (node->left != NULL) queue[back++] = node->left;
            if (node->right != NULL) queue[back++] = node->right;
        }

        (*returnSize)++;
    }

    free(queue);
    return result;
}
```

---

## Pattern 9B: Right Side View

```c
int* rightSideView(struct TreeNode* root, int* returnSize) {
    if (root == NULL) {
        *returnSize = 0;
        return NULL;
    }

    int* result = (int*)malloc(1000 * sizeof(int));
    *returnSize = 0;

    struct TreeNode** queue = (struct TreeNode**)malloc(10000 * sizeof(struct TreeNode*));
    int front = 0, back = 0;
    queue[back++] = root;

    while (front < back) {
        int size = back - front;

        for (int i = 0; i < size; i++) {
            struct TreeNode* node = queue[front++];
            if (i == size - 1) result[(*returnSize)++] = node->val;
            if (node->left != NULL) queue[back++] = node->left;
            if (node->right != NULL) queue[back++] = node->right;
        }
    }

    free(queue);
    return result;
}
```

---

# 10. GRAPHS - DFS

---

## Pattern 10A: Flood Fill / Connected Components

### Problem: Number of Islands

```c
void dfs(char** grid, int rows, int cols, int r, int c) {
    if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] == '0') {
        return;
    }

    grid[r][c] = '0';
    dfs(grid, rows, cols, r + 1, c);
    dfs(grid, rows, cols, r - 1, c);
    dfs(grid, rows, cols, r, c + 1);
    dfs(grid, rows, cols, r, c - 1);
}

int numIslands(char** grid, int gridSize, int* gridColSize) {
    if (gridSize == 0) return 0;

    int rows = gridSize, cols = gridColSize[0];
    int count = 0;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (grid[r][c] == '1') {
                count++;
                dfs(grid, rows, cols, r, c);
            }
        }
    }

    return count;
}
```

---

# 11. GRAPHS - BFS (SHORTEST PATH)

---

## Pattern 11A: Shortest Path in Unweighted Graph

```c
int shortestPathBinaryMatrix(int** grid, int gridSize, int* gridColSize) {
    int n = gridSize;
    if (grid[0][0] == 1 || grid[n-1][n-1] == 1) return -1;

    int dr[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    int dc[] = {-1, 0, 1, -1, 1, -1, 0, 1};

    // Queue: [row, col, dist]
    int* queueR = (int*)malloc(n * n * sizeof(int));
    int* queueC = (int*)malloc(n * n * sizeof(int));
    int* queueD = (int*)malloc(n * n * sizeof(int));
    int front = 0, back = 0;

    queueR[back] = 0;
    queueC[back] = 0;
    queueD[back] = 1;
    back++;
    grid[0][0] = 1;

    while (front < back) {
        int r = queueR[front];
        int c = queueC[front];
        int dist = queueD[front];
        front++;

        if (r == n - 1 && c == n - 1) {
            free(queueR);
            free(queueC);
            free(queueD);
            return dist;
        }

        for (int d = 0; d < 8; d++) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr >= 0 && nr < n && nc >= 0 && nc < n && grid[nr][nc] == 0) {
                grid[nr][nc] = 1;
                queueR[back] = nr;
                queueC[back] = nc;
                queueD[back] = dist + 1;
                back++;
            }
        }
    }

    free(queueR);
    free(queueC);
    free(queueD);
    return -1;
}
```

---

# 12. TOPOLOGICAL SORT

---

## Pattern 12A: Course Schedule (Cycle Detection)

```c
bool canFinish(int numCourses, int** prerequisites, int prerequisitesSize, int* prerequisitesColSize) {
    // Build adjacency list and indegree array
    int* indegree = (int*)calloc(numCourses, sizeof(int));
    int** graph = (int**)malloc(numCourses * sizeof(int*));
    int* graphSize = (int*)calloc(numCourses, sizeof(int));

    for (int i = 0; i < numCourses; i++) {
        graph[i] = (int*)malloc(numCourses * sizeof(int));
    }

    for (int i = 0; i < prerequisitesSize; i++) {
        int from = prerequisites[i][1];
        int to = prerequisites[i][0];
        graph[from][graphSize[from]++] = to;
        indegree[to]++;
    }

    // Queue
    int* queue = (int*)malloc(numCourses * sizeof(int));
    int front = 0, back = 0;

    for (int i = 0; i < numCourses; i++) {
        if (indegree[i] == 0) queue[back++] = i;
    }

    int completed = 0;
    while (front < back) {
        int course = queue[front++];
        completed++;

        for (int i = 0; i < graphSize[course]; i++) {
            int next = graph[course][i];
            indegree[next]--;
            if (indegree[next] == 0) queue[back++] = next;
        }
    }

    // Free memory
    for (int i = 0; i < numCourses; i++) free(graph[i]);
    free(graph);
    free(graphSize);
    free(indegree);
    free(queue);

    return completed == numCourses;
}
```

---

## Pattern 12B: Course Schedule II (Return Order)

```c
int* findOrder(int numCourses, int** prerequisites, int prerequisitesSize, int* prerequisitesColSize, int* returnSize) {
    int* indegree = (int*)calloc(numCourses, sizeof(int));
    int** graph = (int**)malloc(numCourses * sizeof(int*));
    int* graphSize = (int*)calloc(numCourses, sizeof(int));

    for (int i = 0; i < numCourses; i++) {
        graph[i] = (int*)malloc(numCourses * sizeof(int));
    }

    for (int i = 0; i < prerequisitesSize; i++) {
        int from = prerequisites[i][1];
        int to = prerequisites[i][0];
        graph[from][graphSize[from]++] = to;
        indegree[to]++;
    }

    int* queue = (int*)malloc(numCourses * sizeof(int));
    int front = 0, back = 0;

    for (int i = 0; i < numCourses; i++) {
        if (indegree[i] == 0) queue[back++] = i;
    }

    int* order = (int*)malloc(numCourses * sizeof(int));
    int idx = 0;

    while (front < back) {
        int course = queue[front++];
        order[idx++] = course;

        for (int i = 0; i < graphSize[course]; i++) {
            int next = graph[course][i];
            indegree[next]--;
            if (indegree[next] == 0) queue[back++] = next;
        }
    }

    // Free memory
    for (int i = 0; i < numCourses; i++) free(graph[i]);
    free(graph);
    free(graphSize);
    free(indegree);
    free(queue);

    if (idx == numCourses) {
        *returnSize = numCourses;
        return order;
    } else {
        *returnSize = 0;
        free(order);
        return NULL;
    }
}
```

---

# 13. HEAPS / PRIORITY QUEUE

---

## Pattern 13A: Kth Largest Element

```c
// Min heap implementation
void heapifyUp(int* heap, int index) {
    while (index > 0) {
        int parent = (index - 1) / 2;
        if (heap[parent] > heap[index]) {
            int temp = heap[parent];
            heap[parent] = heap[index];
            heap[index] = temp;
            index = parent;
        } else {
            break;
        }
    }
}

void heapifyDown(int* heap, int size, int index) {
    while (2 * index + 1 < size) {
        int smallest = index;
        int left = 2 * index + 1;
        int right = 2 * index + 2;

        if (heap[left] < heap[smallest]) smallest = left;
        if (right < size && heap[right] < heap[smallest]) smallest = right;

        if (smallest != index) {
            int temp = heap[index];
            heap[index] = heap[smallest];
            heap[smallest] = temp;
            index = smallest;
        } else {
            break;
        }
    }
}

int findKthLargest(int* nums, int numsSize, int k) {
    int* minHeap = (int*)malloc(k * sizeof(int));
    int heapSize = 0;

    for (int i = 0; i < numsSize; i++) {
        if (heapSize < k) {
            minHeap[heapSize++] = nums[i];
            heapifyUp(minHeap, heapSize - 1);
        } else if (nums[i] > minHeap[0]) {
            minHeap[0] = nums[i];
            heapifyDown(minHeap, heapSize, 0);
        }
    }

    int result = minHeap[0];
    free(minHeap);
    return result;
}
```

---

## Pattern 13B: Merge K Sorted Lists

```c
// Using min heap of list nodes
struct ListNode* mergeKLists(struct ListNode** lists, int listsSize) {
    if (listsSize == 0) return NULL;

    // Min heap storing pointers
    struct ListNode** heap = (struct ListNode**)malloc(listsSize * sizeof(struct ListNode*));
    int heapSize = 0;

    // Initialize heap with non-null heads
    for (int i = 0; i < listsSize; i++) {
        if (lists[i] != NULL) {
            heap[heapSize++] = lists[i];
        }
    }

    if (heapSize == 0) {
        free(heap);
        return NULL;
    }

    // Build min heap
    for (int i = heapSize / 2 - 1; i >= 0; i--) {
        // heapify down for ListNode pointers
        int index = i;
        while (2 * index + 1 < heapSize) {
            int smallest = index;
            int left = 2 * index + 1;
            int right = 2 * index + 2;

            if (heap[left]->val < heap[smallest]->val) smallest = left;
            if (right < heapSize && heap[right]->val < heap[smallest]->val) smallest = right;

            if (smallest != index) {
                struct ListNode* temp = heap[index];
                heap[index] = heap[smallest];
                heap[smallest] = temp;
                index = smallest;
            } else {
                break;
            }
        }
    }

    struct ListNode dummy;
    dummy.next = NULL;
    struct ListNode* curr = &dummy;

    while (heapSize > 0) {
        struct ListNode* node = heap[0];
        curr->next = node;
        curr = curr->next;

        if (node->next != NULL) {
            heap[0] = node->next;
        } else {
            heap[0] = heap[--heapSize];
        }

        // Heapify down
        int index = 0;
        while (2 * index + 1 < heapSize) {
            int smallest = index;
            int left = 2 * index + 1;
            int right = 2 * index + 2;

            if (heap[left]->val < heap[smallest]->val) smallest = left;
            if (right < heapSize && heap[right]->val < heap[smallest]->val) smallest = right;

            if (smallest != index) {
                struct ListNode* temp = heap[index];
                heap[index] = heap[smallest];
                heap[smallest] = temp;
                index = smallest;
            } else {
                break;
            }
        }
    }

    free(heap);
    return dummy.next;
}
```

---

# 14. BACKTRACKING

---

## Pattern 14A: Subsets

```c
void backtrack(int* nums, int numsSize, int start, int* current, int currentSize,
               int** result, int* returnSize, int** returnColumnSizes) {
    // Add current subset to result
    result[*returnSize] = (int*)malloc(currentSize * sizeof(int));
    memcpy(result[*returnSize], current, currentSize * sizeof(int));
    (*returnColumnSizes)[*returnSize] = currentSize;
    (*returnSize)++;

    for (int i = start; i < numsSize; i++) {
        current[currentSize] = nums[i];
        backtrack(nums, numsSize, i + 1, current, currentSize + 1, result, returnSize, returnColumnSizes);
    }
}

int** subsets(int* nums, int numsSize, int* returnSize, int** returnColumnSizes) {
    int maxSize = 1 << numsSize;  // 2^n subsets
    int** result = (int**)malloc(maxSize * sizeof(int*));
    *returnColumnSizes = (int*)malloc(maxSize * sizeof(int));
    *returnSize = 0;

    int* current = (int*)malloc(numsSize * sizeof(int));
    backtrack(nums, numsSize, 0, current, 0, result, returnSize, returnColumnSizes);

    free(current);
    return result;
}
```

---

## Pattern 14B: Permutations

```c
void backtrackPermute(int* nums, int numsSize, int* current, int currentSize, bool* used,
                      int** result, int* returnSize, int** returnColumnSizes) {
    if (currentSize == numsSize) {
        result[*returnSize] = (int*)malloc(numsSize * sizeof(int));
        memcpy(result[*returnSize], current, numsSize * sizeof(int));
        (*returnColumnSizes)[*returnSize] = numsSize;
        (*returnSize)++;
        return;
    }

    for (int i = 0; i < numsSize; i++) {
        if (used[i]) continue;

        used[i] = true;
        current[currentSize] = nums[i];
        backtrackPermute(nums, numsSize, current, currentSize + 1, used, result, returnSize, returnColumnSizes);
        used[i] = false;
    }
}

int** permute(int* nums, int numsSize, int* returnSize, int** returnColumnSizes) {
    // n! permutations
    int maxSize = 1;
    for (int i = 1; i <= numsSize; i++) maxSize *= i;

    int** result = (int**)malloc(maxSize * sizeof(int*));
    *returnColumnSizes = (int*)malloc(maxSize * sizeof(int));
    *returnSize = 0;

    int* current = (int*)malloc(numsSize * sizeof(int));
    bool* used = (bool*)calloc(numsSize, sizeof(bool));

    backtrackPermute(nums, numsSize, current, 0, used, result, returnSize, returnColumnSizes);

    free(current);
    free(used);
    return result;
}
```

---

## Pattern 14C: Combination Sum

```c
void backtrackCombination(int* candidates, int candidatesSize, int remaining, int start,
                          int* current, int currentSize, int** result, int* returnSize, int** returnColumnSizes) {
    if (remaining == 0) {
        result[*returnSize] = (int*)malloc(currentSize * sizeof(int));
        memcpy(result[*returnSize], current, currentSize * sizeof(int));
        (*returnColumnSizes)[*returnSize] = currentSize;
        (*returnSize)++;
        return;
    }
    if (remaining < 0) return;

    for (int i = start; i < candidatesSize; i++) {
        current[currentSize] = candidates[i];
        backtrackCombination(candidates, candidatesSize, remaining - candidates[i], i,
                             current, currentSize + 1, result, returnSize, returnColumnSizes);
    }
}

int** combinationSum(int* candidates, int candidatesSize, int target, int* returnSize, int** returnColumnSizes) {
    int** result = (int**)malloc(1000 * sizeof(int*));
    *returnColumnSizes = (int*)malloc(1000 * sizeof(int));
    *returnSize = 0;

    int* current = (int*)malloc(target * sizeof(int));

    backtrackCombination(candidates, candidatesSize, target, 0, current, 0, result, returnSize, returnColumnSizes);

    free(current);
    return result;
}
```

---

## Pattern 14D: Word Search

```c
bool backtrackWord(char** board, int rows, int cols, char* word, int r, int c, int index) {
    if (word[index] == '\0') return true;

    if (r < 0 || r >= rows || c < 0 || c >= cols || board[r][c] != word[index]) {
        return false;
    }

    char temp = board[r][c];
    board[r][c] = '#';

    bool found = backtrackWord(board, rows, cols, word, r + 1, c, index + 1) ||
                 backtrackWord(board, rows, cols, word, r - 1, c, index + 1) ||
                 backtrackWord(board, rows, cols, word, r, c + 1, index + 1) ||
                 backtrackWord(board, rows, cols, word, r, c - 1, index + 1);

    board[r][c] = temp;

    return found;
}

bool exist(char** board, int boardSize, int* boardColSize, char* word) {
    int rows = boardSize, cols = boardColSize[0];

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (backtrackWord(board, rows, cols, word, r, c, 0)) {
                return true;
            }
        }
    }

    return false;
}
```

---

## Pattern 14E: N-Queens

```c
void backtrackQueens(int n, int row, int* cols, int* posDiag, int* negDiag,
                     char** board, char*** result, int* returnSize, int** returnColumnSizes) {
    if (row == n) {
        result[*returnSize] = (char**)malloc(n * sizeof(char*));
        for (int i = 0; i < n; i++) {
            result[*returnSize][i] = (char*)malloc((n + 1) * sizeof(char));
            strcpy(result[*returnSize][i], board[i]);
        }
        (*returnColumnSizes)[*returnSize] = n;
        (*returnSize)++;
        return;
    }

    for (int col = 0; col < n; col++) {
        if (cols[col] || posDiag[row + col] || negDiag[row - col + n]) {
            continue;
        }

        cols[col] = 1;
        posDiag[row + col] = 1;
        negDiag[row - col + n] = 1;
        board[row][col] = 'Q';

        backtrackQueens(n, row + 1, cols, posDiag, negDiag, board, result, returnSize, returnColumnSizes);

        cols[col] = 0;
        posDiag[row + col] = 0;
        negDiag[row - col + n] = 0;
        board[row][col] = '.';
    }
}

char*** solveNQueens(int n, int* returnSize, int** returnColumnSizes) {
    char*** result = (char***)malloc(1000 * sizeof(char**));
    *returnColumnSizes = (int*)malloc(1000 * sizeof(int));
    *returnSize = 0;

    char** board = (char**)malloc(n * sizeof(char*));
    for (int i = 0; i < n; i++) {
        board[i] = (char*)malloc((n + 1) * sizeof(char));
        for (int j = 0; j < n; j++) board[i][j] = '.';
        board[i][n] = '\0';
    }

    int* cols = (int*)calloc(n, sizeof(int));
    int* posDiag = (int*)calloc(2 * n, sizeof(int));
    int* negDiag = (int*)calloc(2 * n, sizeof(int));

    backtrackQueens(n, 0, cols, posDiag, negDiag, board, result, returnSize, returnColumnSizes);

    for (int i = 0; i < n; i++) free(board[i]);
    free(board);
    free(cols);
    free(posDiag);
    free(negDiag);

    return result;
}
```

---

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
| "kth largest/merge k sorted"    | Heap                                 |
| "all combinations/permutations" | Backtracking                         |

---

# C Syntax Cheat Sheet

```c
// Common includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>

// Dynamic array allocation
int* arr = (int*)malloc(n * sizeof(int));
int* arr = (int*)calloc(n, sizeof(int));  // zero-initialized
arr = (int*)realloc(arr, newSize * sizeof(int));
free(arr);

// 2D array allocation
int** matrix = (int**)malloc(rows * sizeof(int*));
for (int i = 0; i < rows; i++) {
    matrix[i] = (int*)malloc(cols * sizeof(int));
}

// String operations
int len = strlen(s);
strcpy(dest, src);
strncpy(dest, src, n);
strcmp(s1, s2);  // returns 0 if equal
strcat(dest, src);
char* substr = s + startIndex;  // pointer arithmetic

// Sorting
int compare(const void* a, const void* b) {
    return *(int*)a - *(int*)b;
}
qsort(arr, n, sizeof(int), compare);

// Min/Max macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Memory operations
memset(arr, 0, n * sizeof(int));  // set to 0
memcpy(dest, src, n * sizeof(int));  // copy

// Common data structure definitions
struct ListNode {
    int val;
    struct ListNode* next;
};

struct TreeNode {
    int val;
    struct TreeNode* left;
    struct TreeNode* right;
};

// Stack using array
int stack[1000];
int top = -1;
stack[++top] = value;  // push
int val = stack[top--];  // pop
int peek = stack[top];
bool empty = (top == -1);

// Queue using array
int queue[1000];
int front = 0, back = 0;
queue[back++] = value;  // enqueue
int val = queue[front++];  // dequeue
bool empty = (front == back);

// Simple hash function
int hash(int key, int size) {
    return ((key % size) + size) % size;
}
```
