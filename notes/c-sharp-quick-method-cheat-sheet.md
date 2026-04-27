# C# Data Structures Cheat Sheet

---

## 1. Array — Fixed size, fast access

```csharp
// Declaration
int[] nums = new int[5];              // [0, 0, 0, 0, 0]
int[] nums2 = { 1, 2, 3, 4, 5 };     // initialised
string[] names = new string[] { "Adam", "Ben" };

// Access & modify
nums2[0] = 10;           // set by index — O(1)
int val = nums2[2];      // get by index — O(1)
int len = nums2.Length;   // property, not method

// Iterate
for (int i = 0; i < nums2.Length; i++) { }
foreach (int n in nums2) { }

// Key methods
Array.Sort(nums2);                    // in-place sort
Array.Reverse(nums2);                 // in-place reverse
int idx = Array.IndexOf(nums2, 3);    // first occurrence, -1 if missing
Array.Fill(nums, 7);                  // fill all elements with 7
bool exists = Array.Exists(nums2, x => x > 3); // predicate search

// 2D array
int[,] grid = new int[3, 3];
grid[0, 1] = 5;
```

**When to use:** Size is known at compile time, need fastest possible access by index, matrices/grids.

---

## 2. List\<T\> — Dynamic array

```csharp
// Create
List<int> list = new List<int>();
List<int> list2 = new List<int> { 1, 2, 3 };

// Add
list.Add(10);                 // append — O(1) amortised
list.AddRange(new[] { 20, 30 }); // append multiple
list.Insert(0, 5);            // insert at index — O(n)

// Access
int first = list[0];          // index access — O(1)
int count = list.Count;       // current size (not .Length)

// Remove
list.Remove(10);              // remove first occurrence — O(n)
list.RemoveAt(0);             // remove by index — O(n)
list.RemoveAll(x => x > 15); // remove all matching predicate

// Search
bool has = list.Contains(5);         // O(n)
int idx = list.IndexOf(5);           // first index, -1 if missing
int found = list.Find(x => x > 10); // first match
List<int> all = list.FindAll(x => x > 10); // all matches
bool any = list.Any(x => x > 10);   // LINQ — needs using System.Linq

// Sort & reverse
list.Sort();                         // in-place sort
list.Sort((a, b) => b.CompareTo(a)); // custom comparator (descending)
list.Reverse();

// Convert
int[] arr = list.ToArray();
```

**When to use:** Default go-to collection. Use whenever you need a resizable array.

---

## 3. Dictionary\<TKey, TValue\> — Key-value lookup

```csharp
// Create
var dict = new Dictionary<string, int>();
var dict2 = new Dictionary<string, int>
{
    { "apple", 3 },
    { "banana", 5 }
};

// Add & update
dict["orange"] = 7;            // add or overwrite — O(1)
dict.Add("grape", 2);          // add only — throws if key exists
dict["orange"] = 10;           // update

// Access
int val = dict["apple"];       // throws KeyNotFoundException if missing
bool got = dict.TryGetValue("apple", out int result); // safe access — use this

// Remove
dict.Remove("banana");         // returns bool

// Check
bool hasKey = dict.ContainsKey("apple");   // O(1)
bool hasVal = dict.ContainsValue(3);       // O(n) — avoid in hot paths
int count = dict.Count;

// Iterate
foreach (KeyValuePair<string, int> kv in dict)
{
    Console.WriteLine($"{kv.Key}: {kv.Value}");
}
foreach (var key in dict.Keys) { }
foreach (var val in dict.Values) { }

// Common pattern: counting frequency
string word = "hello";
var freq = new Dictionary<char, int>();
foreach (char c in word)
{
    if (freq.ContainsKey(c))
        freq[c]++;
    else
        freq[c] = 1;
}

// Cleaner with TryGetValue
foreach (char c in word)
{
    freq.TryGetValue(c, out int current);
    freq[c] = current + 1;
}
```

**When to use:** Fast lookup by key, counting, grouping, caching, mapping relationships.

---

## 4. HashSet\<T\> — Uniqueness and fast lookup

```csharp
// Create
var set = new HashSet<int>();
var set2 = new HashSet<int> { 1, 2, 3, 4, 5 };

// Add & remove
set.Add(10);        // returns false if already exists — O(1)
set.Remove(10);     // returns false if not found — O(1)
set.Clear();

// Check
bool has = set2.Contains(3);    // O(1) — this is the main advantage
int count = set2.Count;

// Set operations
var a = new HashSet<int> { 1, 2, 3 };
var b = new HashSet<int> { 2, 3, 4 };

a.UnionWith(b);        // a = {1, 2, 3, 4}       — OR
a.IntersectWith(b);    // a = {2, 3}              — AND
a.ExceptWith(b);       // a = {1}                 — MINUS
a.SymmetricExceptWith(b); // elements in one but not both

bool isSubset = a.IsSubsetOf(b);
bool overlaps = a.Overlaps(b);

// Common pattern: find duplicates
int[] nums = { 1, 2, 3, 2, 4, 3 };
var seen = new HashSet<int>();
foreach (int n in nums)
{
    if (!seen.Add(n))
        Console.WriteLine($"Duplicate: {n}");
}

// Convert list to unique values
List<int> unique = new HashSet<int>(nums).ToList();
```

**When to use:** Checking if something exists in O(1), removing duplicates, set operations (union/intersection).

---

## 5. Queue\<T\> — FIFO (First In, First Out)

```csharp
// Create
var queue = new Queue<int>();
var queue2 = new Queue<string>(new[] { "a", "b", "c" });

// Add & remove
queue.Enqueue(1);     // add to back — O(1)
queue.Enqueue(2);
queue.Enqueue(3);     // queue: [1, 2, 3]

int front = queue.Dequeue();   // remove from front — O(1) — returns 1
int peek = queue.Peek();       // look at front without removing — returns 2

// Check
int count = queue.Count;
bool has = queue.Contains(3);  // O(n)

// Safe dequeue
if (queue.TryDequeue(out int result)) { }
if (queue.TryPeek(out int peekResult)) { }

// BFS example
var graph = new Dictionary<int, List<int>>
{
    { 0, new List<int> { 1, 2 } },
    { 1, new List<int> { 3 } },
    { 2, new List<int> { 3 } },
    { 3, new List<int>() }
};

var visited = new HashSet<int>();
var bfsQueue = new Queue<int>();
bfsQueue.Enqueue(0);
visited.Add(0);

while (bfsQueue.Count > 0)
{
    int node = bfsQueue.Dequeue();
    Console.WriteLine(node);
    foreach (int neighbour in graph[node])
    {
        if (visited.Add(neighbour))
            bfsQueue.Enqueue(neighbour);
    }
}
```

**When to use:** BFS, level-order traversal, task scheduling, processing things in order.

---

## 6. Stack\<T\> — LIFO (Last In, First Out)

```csharp
// Create
var stack = new Stack<int>();

// Add & remove
stack.Push(1);
stack.Push(2);
stack.Push(3);        // stack top: 3

int top = stack.Pop();    // remove from top — O(1) — returns 3
int peek = stack.Peek();  // look at top without removing — returns 2

// Check
int count = stack.Count;
bool has = stack.Contains(1); // O(n)

// Safe pop
if (stack.TryPop(out int result)) { }
if (stack.TryPeek(out int peekResult)) { }

// Classic example: valid parentheses
bool IsValid(string s)
{
    var stack = new Stack<char>();
    var pairs = new Dictionary<char, char>
    {
        { ')', '(' }, { ']', '[' }, { '}', '{' }
    };

    foreach (char c in s)
    {
        if (pairs.ContainsValue(c))
        {
            stack.Push(c);
        }
        else if (pairs.ContainsKey(c))
        {
            if (stack.Count == 0 || stack.Pop() != pairs[c])
                return false;
        }
    }
    return stack.Count == 0;
}

// DFS with explicit stack
var dfsStack = new Stack<int>();
var visited = new HashSet<int>();
dfsStack.Push(0);

while (dfsStack.Count > 0)
{
    int node = dfsStack.Pop();
    if (!visited.Add(node)) continue;
    Console.WriteLine(node);
    foreach (int neighbour in graph[node])
        dfsStack.Push(neighbour);
}
```

**When to use:** DFS, undo/redo, expression parsing, backtracking, matching brackets.

---

## 7. PriorityQueue\<TElement, TPriority\> — Min-heap (C# 10+)

```csharp
// Create — lowest priority number comes out first (min-heap)
var pq = new PriorityQueue<string, int>();

// Add
pq.Enqueue("low priority task", 10);
pq.Enqueue("urgent task", 1);
pq.Enqueue("medium task", 5);

// Remove — always returns lowest priority value first
string next = pq.Dequeue();     // "urgent task" (priority 1)
string peek = pq.Peek();        // look without removing

// Check
int count = pq.Count;

// Safe access
if (pq.TryDequeue(out string element, out int priority)) { }
if (pq.TryPeek(out string el, out int pri)) { }

// Dijkstra's shortest path example
var adjList = new Dictionary<int, List<(int node, int weight)>>
{
    { 0, new() { (1, 4), (2, 1) } },
    { 1, new() { (3, 1) } },
    { 2, new() { (1, 2), (3, 5) } },
    { 3, new() }
};

var dist = new Dictionary<int, int>();
var dijkstra = new PriorityQueue<int, int>();
dijkstra.Enqueue(0, 0);
dist[0] = 0;

while (dijkstra.Count > 0)
{
    int current = dijkstra.Dequeue();

    foreach (var (neighbour, weight) in adjList[current])
    {
        int newDist = dist[current] + weight;
        if (!dist.ContainsKey(neighbour) || newDist < dist[neighbour])
        {
            dist[neighbour] = newDist;
            dijkstra.Enqueue(neighbour, newDist);
        }
    }
}

// NOTE: For a max-heap, negate the priority
var maxHeap = new PriorityQueue<string, int>();
maxHeap.Enqueue("small", -1);
maxHeap.Enqueue("large", -100);
// Dequeue returns "large" first (priority -100 < -1)
```

**When to use:** Dijkstra's algorithm, task scheduling by priority, K-th largest/smallest, merge K sorted lists.

---

## Quick Comparison

| Structure     | Access     | Search   | Insert        | Delete   | Ordered?    |
| ------------- | ---------- | -------- | ------------- | -------- | ----------- |
| Array         | O(1)       | O(n)     | N/A           | N/A      | Yes (index) |
| List\<T\>     | O(1)       | O(n)     | O(1)\* / O(n) | O(n)     | Yes (index) |
| Dictionary    | O(1)       | O(1) key | O(1)          | O(1)     | No          |
| HashSet       | —          | O(1)     | O(1)          | O(1)     | No          |
| Queue         | Front O(1) | O(n)     | O(1)          | O(1)     | FIFO        |
| Stack         | Top O(1)   | O(n)     | O(1)          | O(1)     | LIFO        |
| PriorityQueue | Min O(1)   | O(n)     | O(log n)      | O(log n) | By priority |

\*List Insert is O(1) amortised for Add (append), O(n) for Insert at index.

---

## Bonus: LinkedList\<T\>

```csharp
var ll = new LinkedList<int>();
ll.AddLast(1);              // append
ll.AddFirst(0);             // prepend
ll.AddAfter(ll.First, 5);  // insert after node

int first = ll.First.Value;
int last = ll.Last.Value;

ll.Remove(5);
ll.RemoveFirst();
ll.RemoveLast();
```

**When to use:** Frequent insertions/deletions in the middle, implementing LRU cache.

---

## Bonus: SortedDictionary\<TKey, TValue\> & SortedSet\<T\>

```csharp
// Like Dictionary but keys are always sorted (backed by red-black tree)
var sd = new SortedDictionary<string, int>
{
    { "banana", 2 }, { "apple", 1 }, { "cherry", 3 }
};
// Iteration order: apple, banana, cherry

// Like HashSet but elements always sorted
var ss = new SortedSet<int> { 5, 1, 3, 2, 4 };
int min = ss.Min;   // 1
int max = ss.Max;   // 5
var range = ss.GetViewBetween(2, 4); // {2, 3, 4}
```

**When to use:** Need sorted keys/elements with O(log n) operations. Sliding window problems with ordered access.
