# Complete DSA Problem-to-Keyword Mapping

## Master Reference Table

| Category             | Pattern Type                       | Problem                                   | Keywords                                                                         | Core Intuition                                          |
| -------------------- | ---------------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------- |
| **Arrays & Hashing** | Complement Lookup                  | Two Sum                                   | "two numbers add to", "pair equals target", "find indices", "sum to target"      | Store seen values; check if complement exists           |
|                      | Frequency Counting                 | Top K Frequent Elements                   | "most frequent", "top k", "count occurrences", "frequency"                       | HashMap to count + bucket sort or heap                  |
|                      | Frequency Counting                 | Valid Anagram                             | "anagram", "same characters", "rearrange"                                        | Compare character frequency counts                      |
|                      | Key Transformation                 | Group Anagrams                            | "group by", "anagrams", "same characters", "categorize strings"                  | Sorted string or char count as hashmap key              |
|                      | Prefix Sum + HashMap               | Subarray Sum Equals K                     | "subarray sum equals", "contiguous sum", "count subarrays"                       | prefix[j] - prefix[i] = k means subarray sums to k      |
|                      | Prefix Sum + HashMap               | Continuous Subarray Sum                   | "subarray sum multiple of k", "divisible by k"                                   | Store prefix_sum % k in hashmap                         |
|                      | Set for Existence                  | Contains Duplicate                        | "contains duplicate", "any duplicates", "repeated element"                       | HashSet to track seen elements                          |
|                      | Set for Lookup                     | Longest Consecutive Sequence              | "longest consecutive", "sequence", "consecutive numbers"                         | HashSet + only start counting from sequence start       |
|                      |                                    |                                           |                                                                                  |                                                         |
| **Two Pointers**     | Opposite Ends                      | Two Sum II (Sorted)                       | "sorted array", "two numbers sum to", "pair in sorted"                           | Left/right pointers, move based on sum comparison       |
|                      | Opposite Ends                      | Container With Most Water                 | "two lines", "container", "maximum area", "most water"                           | Start widest, move shorter pointer inward               |
|                      | Opposite Ends                      | Trapping Rain Water                       | "trap water", "rain water", "elevation map"                                      | Track max heights from both ends                        |
|                      | Opposite Ends + Fixed First        | 3Sum                                      | "three numbers sum to", "triplets", "unique triplets"                            | Sort + fix one + two pointers for remaining             |
|                      | Opposite Ends + Fixed First        | 4Sum                                      | "four numbers sum to", "quadruplets"                                             | Sort + fix two + two pointers for remaining             |
|                      | Opposite Ends                      | Valid Palindrome                          | "palindrome", "reads same backwards", "ignore non-alphanumeric"                  | Compare from both ends moving inward                    |
|                      | Slow/Fast Same Direction           | Remove Duplicates from Sorted Array       | "remove duplicates", "in-place", "sorted array", "unique elements"               | Slow = write position, fast = reader                    |
|                      | Slow/Fast Same Direction           | Move Zeroes                               | "move zeroes", "maintain order", "in-place", "zeroes to end"                     | Swap non-zeros to front                                 |
|                      | Slow/Fast Same Direction           | Remove Element                            | "remove element", "in-place", "remove value"                                     | Skip unwanted, write wanted                             |
|                      | Two Arrays                         | Merge Sorted Array                        | "merge sorted", "two sorted arrays", "merge in-place"                            | Fill from end to avoid overwriting                      |
|                      |                                    |                                           |                                                                                  |                                                         |
| **Sliding Window**   | Fixed Size                         | Maximum Sum Subarray of Size K            | "subarray of size k", "window of size k", "consecutive k elements"               | Add new element, remove old, track max                  |
|                      | Fixed Size                         | Find All Anagrams                         | "find anagrams", "permutation in string", "sliding anagram"                      | Fixed window of pattern length, compare counts          |
|                      | Fixed Size                         | Permutation in String                     | "permutation exists", "rearrangement in string"                                  | Sliding window with character count match               |
|                      | Variable - Longest Valid           | Longest Substring Without Repeating       | "longest substring", "without repeating", "no duplicates", "distinct characters" | Expand right, shrink left when duplicate found          |
|                      | Variable - Longest Valid           | Longest Substring with At Most K Distinct | "at most k distinct", "k unique characters"                                      | Shrink when distinct count exceeds k                    |
|                      | Variable - Longest with Constraint | Longest Repeating Character Replacement   | "at most k replacements", "k changes allowed", "make all same"                   | Valid if (window_size - max_freq) ≤ k                   |
|                      | Variable - Longest with Constraint | Max Consecutive Ones III                  | "flip at most k zeroes", "consecutive ones", "k flips"                           | Count zeros in window, shrink if > k                    |
|                      | Variable - Shortest Valid          | Minimum Window Substring                  | "minimum window", "shortest substring containing", "smallest with all chars"     | Expand until valid, shrink while still valid            |
|                      | Variable - Shortest Valid          | Minimum Size Subarray Sum                 | "minimum size", "sum at least", "shortest subarray ≥ target"                     | Shrink while sum still meets condition                  |
|                      |                                    |                                           |                                                                                  |                                                         |
| **Binary Search**    | Standard                           | Binary Search                             | "sorted array", "find element", "O(log n)", "search in sorted"                   | Compare mid, eliminate half                             |
|                      | Standard                           | Search Insert Position                    | "insert position", "where to insert", "sorted array"                             | Find leftmost position ≥ target                         |
|                      | Find Boundary                      | First and Last Position                   | "first occurrence", "last occurrence", "range of target"                         | Two binary searches for left and right bounds           |
|                      | Find Boundary                      | Find Smallest Letter Greater Than Target  | "smallest greater", "next letter"                                                | Find first element > target                             |
|                      | Rotated Array                      | Search in Rotated Sorted Array            | "rotated sorted", "pivot", "rotated array", "search rotated"                     | One half always sorted, determine which                 |
|                      | Rotated Array                      | Find Minimum in Rotated Sorted Array      | "minimum in rotated", "find pivot", "smallest element"                           | Binary search toward unsorted half                      |
|                      | Search on Answer                   | Koko Eating Bananas                       | "minimum speed", "finish in time", "minimum rate"                                | Binary search on speed, check if feasible               |
|                      | Search on Answer                   | Capacity to Ship Packages                 | "minimum capacity", "ship within days", "minimum weight limit"                   | Binary search on capacity                               |
|                      | Search on Answer                   | Split Array Largest Sum                   | "split into k parts", "minimize largest sum"                                     | Binary search on max subarray sum                       |
|                      | Peak Finding                       | Find Peak Element                         | "peak element", "local maximum", "greater than neighbors"                        | Move toward higher neighbor                             |
|                      | 2D Search                          | Search 2D Matrix                          | "search matrix", "sorted rows and columns"                                       | Treat as 1D array or staircase search                   |
|                      |                                    |                                           |                                                                                  |                                                         |
| **Stacks**           | Matching Pairs                     | Valid Parentheses                         | "valid parentheses", "balanced brackets", "matching pairs", "properly nested"    | Push open, pop and match on close                       |
|                      | Matching Pairs                     | Minimum Remove to Make Valid              | "remove invalid", "make valid parentheses", "minimum removals"                   | Track unmatched indices                                 |
|                      | Monotonic Decreasing               | Daily Temperatures                        | "next greater", "next warmer", "days until warmer", "wait for higher"            | Stack holds waiting elements, pop when resolved         |
|                      | Monotonic Decreasing               | Next Greater Element                      | "next greater", "first larger to right"                                          | Pop smaller elements when larger arrives                |
|                      | Monotonic Increasing               | Next Smaller Element                      | "next smaller", "first smaller to right"                                         | Pop larger elements when smaller arrives                |
|                      | Monotonic                          | Largest Rectangle in Histogram            | "largest rectangle", "histogram", "max rectangle area"                           | Find left/right boundaries for each bar                 |
|                      | Monotonic                          | Maximal Rectangle (in matrix)             | "maximal rectangle", "largest rectangle of 1s"                                   | Build histogram per row, apply largest rectangle        |
|                      | Design                             | Min Stack                                 | "min in O(1)", "getMin constant", "design stack with min"                        | Parallel stack tracking minimum                         |
|                      | Evaluation                         | Evaluate Reverse Polish Notation          | "reverse polish", "postfix", "evaluate expression"                               | Push operands, pop and compute on operators             |
|                      | Simplification                     | Simplify Path                             | "simplify path", "canonical path", "unix path"                                   | Stack for directory names, handle . and ..              |
|                      | Decoding                           | Decode String                             | "decode", "k[string]", "nested encoding"                                         | Stack for multipliers and partial results               |
|                      |                                    |                                           |                                                                                  |                                                         |
| **Queues**           | BFS Level by Level                 | Rotting Oranges                           | "spreads", "minimum time", "infection spreads", "simultaneous spread"            | Multi-source BFS, each level = 1 time unit              |
|                      | BFS Level by Level                 | Walls and Gates                           | "distance to nearest", "fill distances", "spread from targets"                   | BFS from all targets simultaneously                     |
|                      | BFS Shortest Path                  | Shortest Path in Binary Matrix            | "shortest path", "minimum steps", "fewest moves in grid"                         | BFS guarantees shortest in unweighted graph             |
|                      | Monotonic Deque                    | Sliding Window Maximum                    | "sliding window maximum", "max of each window", "max in k elements"              | Deque stores indices in decreasing value order          |
|                      | Monotonic Deque                    | Sliding Window Minimum                    | "sliding window minimum", "min of each window"                                   | Deque stores indices in increasing value order          |
|                      | Design                             | Implement Queue using Stacks              | "queue using stacks", "implement queue"                                          | Two stacks: one for push, one for pop                   |
|                      | Design                             | Implement Stack using Queues              | "stack using queues", "implement stack"                                          | Rotate queue on each push                               |
|                      |                                    |                                           |                                                                                  |                                                         |
| **Linked Lists**     | Pointer Reversal                   | Reverse Linked List                       | "reverse", "flip list", "reverse pointers", "reverse order"                      | prev/curr/next pointer manipulation                     |
|                      | Pointer Reversal                   | Reverse Linked List II                    | "reverse between", "reverse portion", "reverse from m to n"                      | Isolate segment, reverse, reconnect                     |
|                      | Pointer Reversal                   | Reverse Nodes in K-Group                  | "reverse in groups", "k nodes at a time"                                         | Check k nodes exist, reverse group, recurse             |
|                      | Fast/Slow                          | Linked List Cycle                         | "cycle", "loop", "circular", "detect cycle", "has loop"                          | Fast catches slow if cycle exists                       |
|                      | Fast/Slow                          | Linked List Cycle II                      | "cycle start", "where cycle begins", "find cycle entry"                          | After meeting, reset one pointer to head                |
|                      | Fast/Slow                          | Middle of Linked List                     | "middle", "halfway", "center node", "mid point"                                  | When fast reaches end, slow is at middle                |
|                      | Fast/Slow                          | Palindrome Linked List                    | "palindrome list", "same forwards and backwards"                                 | Find middle, reverse second half, compare               |
|                      | Two Pointers with Gap              | Remove Nth From End                       | "nth from end", "kth from end", "remove from back"                               | Create n-gap between pointers                           |
|                      | Two Pointers with Gap              | Intersection of Two Lists                 | "intersection", "where lists meet", "common node"                                | Align by length or cycle through both                   |
|                      | Merge                              | Merge Two Sorted Lists                    | "merge sorted lists", "combine two lists"                                        | Compare heads, append smaller                           |
|                      | Merge                              | Merge K Sorted Lists                      | "merge k lists", "combine k sorted"                                              | Heap of k heads, extract min                            |
|                      | Dummy Node                         | Add Two Numbers                           | "add numbers", "sum as linked list", "digit by digit"                            | Dummy head, handle carry                                |
|                      | Dummy Node                         | Remove Duplicates from Sorted List        | "remove duplicates", "sorted list", "unique nodes"                               | Skip nodes with duplicate values                        |
|                      |                                    |                                           |                                                                                  |                                                         |
| **Trees - DFS**      | Basic Recursion                    | Maximum Depth of Binary Tree              | "depth", "height", "how deep", "max depth"                                       | 1 + max(left_depth, right_depth)                        |
|                      | Basic Recursion                    | Minimum Depth of Binary Tree              | "minimum depth", "shortest path to leaf"                                         | Handle null children carefully                          |
|                      | Basic Recursion                    | Same Tree                                 | "same tree", "identical", "equal trees"                                          | Compare structure and values recursively                |
|                      | Basic Recursion                    | Symmetric Tree                            | "symmetric", "mirror", "left equals right"                                       | Compare left.left with right.right, etc.                |
|                      | Basic Recursion                    | Invert Binary Tree                        | "invert", "flip", "mirror tree"                                                  | Swap left and right recursively                         |
|                      | Path Tracking                      | Path Sum                                  | "path sum", "root to leaf sum", "sum equals target"                              | Track remaining sum to leaves                           |
|                      | Path Tracking                      | Path Sum II                               | "all paths with sum", "find all paths", "collect paths"                          | Backtrack to collect all valid paths                    |
|                      | Path Tracking                      | Binary Tree Maximum Path Sum              | "maximum path sum", "max sum any path"                                           | Track max through node vs max overall                   |
|                      | Validation                         | Validate Binary Search Tree               | "valid BST", "is BST", "BST property"                                            | Pass valid range down, or check inorder ascending       |
|                      | Validation                         | Balanced Binary Tree                      | "balanced", "height balanced", "depths differ by 1"                              | Check heights differ by at most 1                       |
|                      | Ancestor                           | Lowest Common Ancestor                    | "common ancestor", "LCA", "lowest ancestor", "meeting point"                     | Return non-null; if both sides non-null, current is LCA |
|                      | Ancestor                           | LCA of BST                                | "LCA in BST", "common ancestor BST"                                              | Use BST property to go left or right                    |
|                      | Construction                       | Construct from Preorder and Inorder       | "construct tree", "build from traversals"                                        | Preorder gives root, inorder splits left/right          |
|                      | Construction                       | Construct from Inorder and Postorder      | "construct from inorder postorder"                                               | Postorder gives root (from end)                         |
|                      | Serialization                      | Serialize and Deserialize Binary Tree     | "serialize", "deserialize", "encode tree"                                        | Preorder with null markers                              |
|                      |                                    |                                           |                                                                                  |                                                         |
| **Trees - BFS**      | Level Order                        | Binary Tree Level Order Traversal         | "level order", "level by level", "BFS tree", "layers"                            | Queue, process level_size nodes per iteration           |
|                      | Level Order                        | Binary Tree Zigzag Level Order            | "zigzag", "alternating direction", "snake order"                                 | Alternate adding to front/back of level list            |
|                      | Level Order                        | Binary Tree Right Side View               | "right side view", "rightmost each level", "view from right"                     | Take last node of each level                            |
|                      | Level Order                        | Binary Tree Left Side View                | "left side view", "leftmost each level"                                          | Take first node of each level                           |
|                      | Level Order                        | Average of Levels                         | "average per level", "level averages"                                            | Sum and count per level                                 |
|                      | Connection                         | Populating Next Right Pointers            | "next pointer", "connect same level", "right neighbor"                           | BFS or use established next pointers                    |
|                      |                                    |                                           |                                                                                  |                                                         |
| **Graphs - DFS**     | Flood Fill                         | Number of Islands                         | "number of islands", "connected components", "count regions", "count groups"     | Count DFS/BFS starts from unvisited land                |
|                      | Flood Fill                         | Max Area of Island                        | "largest island", "maximum area", "biggest region"                               | Track size during DFS                                   |
|                      | Flood Fill                         | Surrounded Regions                        | "surrounded", "capture regions", "border connected"                              | DFS from borders first, mark safe, then flip rest       |
|                      | Flood Fill                         | Pacific Atlantic Water Flow               | "reach both oceans", "flow to edges", "water flow"                               | DFS from each ocean inward, find intersection           |
|                      | Clone/Copy                         | Clone Graph                               | "clone graph", "deep copy graph", "duplicate graph"                              | HashMap old→new, DFS to clone neighbors                 |
|                      | Cycle Detection                    | Course Schedule (DFS version)             | "cycle in directed graph", "detect cycle DFS"                                    | Three states: unvisited, visiting, visited              |
|                      | Cycle Detection                    | Graph Valid Tree                          | "valid tree", "no cycles", "connected acyclic"                                   | n-1 edges + connected + no cycle                        |
|                      | All Paths                          | All Paths from Source to Target           | "all paths", "find all paths", "source to target"                                | DFS with path tracking, backtrack                       |
|                      | Bipartite                          | Is Graph Bipartite                        | "bipartite", "two coloring", "two groups"                                        | Try 2-coloring with DFS/BFS                             |
|                      |                                    |                                           |                                                                                  |                                                         |
| **Graphs - BFS**     | Shortest Path Unweighted           | Shortest Path in Binary Matrix            | "shortest path", "minimum steps", "fewest moves"                                 | BFS guarantees shortest in unweighted                   |
|                      | Shortest Path Unweighted           | Word Ladder                               | "word transformation", "one letter change", "minimum transformations"            | BFS where neighbors differ by one letter                |
|                      | Shortest Path Unweighted           | Open the Lock                             | "minimum turns", "lock combinations", "shortest to target"                       | BFS on state space (lock combos)                        |
|                      | Multi-Source BFS                   | 01 Matrix                                 | "distance to nearest 0", "closest zero"                                          | Start BFS from all 0s simultaneously                    |
|                      | Multi-Source BFS                   | Rotting Oranges                           | "infection spread", "simultaneous spread", "minimum time to infect all"          | All rotten oranges in queue initially                   |
|                      | Layer by Layer                     | Minimum Knight Moves                      | "knight moves", "chess knight", "minimum moves"                                  | BFS with knight move offsets                            |
|                      |                                    |                                           |                                                                                  |                                                         |
| **Topological Sort** | Cycle Detection (Kahn's)           | Course Schedule                           | "can finish courses", "prerequisites possible", "detect cycle in prerequisites"  | Indegree 0 first; if can't process all, cycle exists    |
|                      | Find Order (Kahn's)                | Course Schedule II                        | "order of courses", "valid sequence", "prerequisite order"                       | Record processing order during Kahn's                   |
|                      | Find Order                         | Alien Dictionary                          | "alien dictionary", "order of characters", "derive alphabet"                     | Build graph from word comparisons, topological sort     |
|                      | Dependency                         | Task Scheduler                            | "task schedule", "cooling period", "minimum intervals"                           | Most frequent task determines minimum time              |
|                      |                                    |                                           |                                                                                  |                                                         |
| **Heaps**            | Kth Element                        | Kth Largest Element in Array              | "kth largest", "kth biggest"                                                     | Min heap of size k, top is answer                       |
|                      | Kth Element                        | Kth Smallest Element in Sorted Matrix     | "kth smallest in matrix"                                                         | Min heap with matrix positions                          |
|                      | Kth Element                        | Kth Largest Element in Stream             | "kth largest stream", "running kth largest"                                      | Maintain min heap of size k                             |
|                      | Top K                              | Top K Frequent Elements                   | "top k frequent", "k most common"                                                | Heap or bucket sort after counting                      |
|                      | Top K                              | Top K Frequent Words                      | "top k words", "most frequent words"                                             | Heap with custom comparator                             |
|                      | K-Way Merge                        | Merge K Sorted Lists                      | "merge k sorted", "combine k lists", "k-way merge"                               | Heap of k list heads                                    |
|                      | K-Way Merge                        | Find K Pairs with Smallest Sums           | "k smallest pairs", "pairs from two arrays"                                      | Heap tracking array indices                             |
|                      | Two Heaps                          | Find Median from Data Stream              | "running median", "stream median", "median online"                               | Max heap for smaller half, min heap for larger half     |
|                      | Two Heaps                          | Sliding Window Median                     | "sliding window median", "median of window"                                      | Two heaps + lazy deletion                               |
|                      | Interval                           | Meeting Rooms II                          | "minimum rooms", "overlapping meetings", "concurrent events"                     | Min heap by end time                                    |
|                      | Interval                           | Task Scheduler                            | "schedule tasks", "idle time", "cooling period"                                  | Max heap for frequencies                                |
|                      | Closest K                          | K Closest Points to Origin                | "k closest", "nearest points", "smallest distance"                               | Max heap of size k by distance                          |
|                      | Closest K                          | Find K Closest Elements                   | "k closest to x", "nearest to target"                                            | Binary search + two pointers or heap                    |
|                      |                                    |                                           |                                                                                  |                                                         |
| **Backtracking**     | Subsets                            | Subsets                                   | "all subsets", "power set", "every combination"                                  | Include/exclude each element                            |
|                      | Subsets                            | Subsets II (with duplicates)              | "unique subsets", "subsets no duplicates"                                        | Sort + skip adjacent duplicates                         |
|                      | Permutations                       | Permutations                              | "all permutations", "all arrangements", "every order"                            | Try each unused element at each position                |
|                      | Permutations                       | Permutations II (with duplicates)         | "unique permutations", "permutations no duplicates"                              | Sort + skip same values at same level                   |
|                      | Combinations                       | Combinations                              | "all combinations", "n choose k", "select k from n"                              | Select k elements with start index                      |
|                      | Combinations                       | Combination Sum                           | "combinations sum to", "sum equals target", "can reuse"                          | Start from i (not i+1) if reuse allowed                 |
|                      | Combinations                       | Combination Sum II (no reuse)             | "combination sum once each", "use once", "each element once"                     | Start from i+1, skip duplicates                         |
|                      | Combinations                       | Combination Sum III                       | "k numbers sum to n", "digits 1-9"                                               | k slots, target sum, digits 1-9                         |
|                      | Grid Search                        | Word Search                               | "word search grid", "find word in matrix", "spell word in grid"                  | DFS with temporary marking                              |
|                      | Grid Search                        | Word Search II                            | "find multiple words", "many words in grid"                                      | Trie + backtracking                                     |
|                      | Constraint Satisfaction            | N-Queens                                  | "n-queens", "queens no attack", "place n queens"                                 | Track columns and diagonals                             |
|                      | Constraint Satisfaction            | Sudoku Solver                             | "solve sudoku", "fill sudoku", "valid sudoku"                                    | Try 1-9 in empty cells, validate                        |
|                      | Partitioning                       | Palindrome Partitioning                   | "partition into palindromes", "split into palindromes"                           | Try each prefix, if palindrome, recurse on rest         |
|                      | Partitioning                       | Restore IP Addresses                      | "restore IP", "valid IP addresses"                                               | Partition into 4 valid segments                         |
|                      | String Generation                  | Generate Parentheses                      | "generate parentheses", "valid parentheses combinations", "n pairs of brackets"  | Track open/close counts, add if valid                   |
|                      | String Generation                  | Letter Combinations of Phone Number       | "phone number letters", "digit to letters", "t9 keyboard"                        | Map digits to letters, combine all                      |

---

## Quick Lookup: Keywords → Pattern

| If You See These Keywords...                  | Think This Pattern                    |
| --------------------------------------------- | ------------------------------------- |
| "two/pair sum to", "find complement"          | HashMap complement lookup             |
| "most frequent", "top k", "count"             | HashMap frequency + heap/bucket       |
| "group by", "anagrams"                        | HashMap with key transformation       |
| "subarray sum equals k"                       | Prefix sum + HashMap                  |
| "sorted + two elements"                       | Two pointers opposite ends            |
| "triplet/3sum/4sum"                           | Sort + fix some + two pointers        |
| "in-place", "remove/move elements"            | Slow/fast same direction              |
| "substring/subarray of size k"                | Fixed sliding window                  |
| "longest substring with..."                   | Variable window (shrink when invalid) |
| "shortest/minimum substring with..."          | Variable window (shrink while valid)  |
| "sorted + find"                               | Binary search                         |
| "minimum X that satisfies condition"          | Binary search on answer               |
| "rotated sorted array"                        | Modified binary search                |
| "valid parentheses", "matching brackets"      | Stack matching                        |
| "next greater/smaller/warmer"                 | Monotonic stack                       |
| "largest rectangle"                           | Monotonic stack                       |
| "spreads/infects", "minimum time to spread"   | Multi-source BFS                      |
| "sliding window max/min"                      | Monotonic deque                       |
| "reverse list"                                | Three pointer reversal                |
| "cycle in list"                               | Fast/slow pointers                    |
| "middle of list"                              | Fast/slow pointers                    |
| "nth from end"                                | Two pointers with gap                 |
| "merge sorted lists"                          | Compare heads technique               |
| "tree depth/height"                           | DFS recursion                         |
| "valid BST"                                   | DFS with range or inorder             |
| "lowest common ancestor"                      | DFS return non-null                   |
| "path sum", "root to leaf"                    | DFS path tracking                     |
| "level order", "level by level"               | BFS with queue                        |
| "right/left side view"                        | BFS take first/last of level          |
| "number of islands", "connected regions"      | DFS/BFS flood fill                    |
| "clone graph"                                 | DFS with hashmap                      |
| "shortest path unweighted"                    | BFS                                   |
| "word transformation"                         | BFS on word graph                     |
| "prerequisites", "dependencies", "task order" | Topological sort                      |
| "can finish all courses"                      | Topological sort (cycle detection)    |
| "kth largest/smallest"                        | Heap of size k                        |
| "merge k sorted"                              | Heap k-way merge                      |
| "running median", "stream median"             | Two heaps                             |
| "all subsets", "power set"                    | Backtracking subsets                  |
| "all permutations", "arrangements"            | Backtracking permutations             |
| "combinations that sum to"                    | Backtracking combination sum          |
| "word search in grid"                         | Backtracking grid DFS                 |
| "n-queens", "sudoku", "no conflicts"          | Backtracking constraint satisfaction  |
| "generate all valid X"                        | Backtracking generation               |

---

## Pattern Subtypes Summary

### Two Pointers Subtypes

| Subtype                   | When to Use                                 | Example                               |
| ------------------------- | ------------------------------------------- | ------------------------------------- |
| Opposite Ends             | Sorted array, find pair, container problems | Two Sum II, Container With Most Water |
| Slow/Fast Same Direction  | In-place modifications, partition           | Remove Duplicates, Move Zeroes        |
| Fast/Slow Different Speed | Cycle detection, middle finding             | Linked List Cycle, Middle of List     |
| Gap Pointers              | Nth from end, intersection                  | Remove Nth From End                   |

### Sliding Window Subtypes

| Subtype           | When to Use               | How to Slide                                 |
| ----------------- | ------------------------- | -------------------------------------------- |
| Fixed Size        | "window of size k"        | Add right, remove left at fixed interval     |
| Variable Longest  | "longest with constraint" | Expand always, shrink when invalid           |
| Variable Shortest | "shortest/minimum valid"  | Expand until valid, shrink while still valid |

### Binary Search Subtypes

| Subtype          | When to Use                | Template                                         |
| ---------------- | -------------------------- | ------------------------------------------------ |
| Standard         | Find exact element         | left <= right, return mid                        |
| Left Boundary    | First occurrence           | Keep going left when found                       |
| Right Boundary   | Last occurrence            | Keep going right when found                      |
| Search on Answer | "minimum X that satisfies" | Binary search on answer space, check feasibility |

### Stack Subtypes

| Subtype              | When to Use           | Stack Content                |
| -------------------- | --------------------- | ---------------------------- |
| Matching             | Brackets, parentheses | Opening brackets             |
| Monotonic Decreasing | Next greater element  | Indices (decreasing values)  |
| Monotonic Increasing | Next smaller element  | Indices (increasing values)  |
| Evaluation           | Expression evaluation | Operands and partial results |

### Heap Subtypes

| Subtype     | When to Use               | Heap Size                |
| ----------- | ------------------------- | ------------------------ |
| Kth Element | Find kth largest/smallest | Fixed size k             |
| Top K       | Find top k elements       | Size k or unbounded      |
| K-Way Merge | Merge k sorted sequences  | Size k (one from each)   |
| Two Heaps   | Running median            | Two heaps splitting data |

### Backtracking Subtypes

| Subtype                   | Start Index Rule | Duplicate Handling             |
| ------------------------- | ---------------- | ------------------------------ |
| Subsets                   | Start from i+1   | Sort + skip adjacent           |
| Permutations              | Check used array | Sort + skip same at same level |
| Combinations (with reuse) | Start from i     | N/A                            |
| Combinations (no reuse)   | Start from i+1   | Sort + skip adjacent           |
