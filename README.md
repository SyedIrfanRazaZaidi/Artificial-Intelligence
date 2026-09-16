# AI Lab - Search Algorithms

NED University of Engineering & Technology
Lab Session 06 - BFS, DFS, and Iterative Deepening Search (IDS)

## Graph used

```
S -> A, D
A -> B, C
D -> B, E
B -> C, E
C -> G
E -> G
```

Start: S | Goal: G | Edge cost: 1 | Heuristic: 0

## Files

- `bfs_dfs.py` - Breadth First Search and Depth First Search
- `ids.py` - Iterative Deepening Search

## Results

| Algorithm | Path | Cost |
|---|---|---|
| BFS | S -> D -> E -> G | 3 |
| DFS | S -> D -> B -> E -> G | 4 |
| IDS | S -> D -> E -> G | 3 |

## Run

```bash
python3 bfs_dfs.py
python3 ids.py
```

## Author

Name: Syed Irfan Raza
Roll No: CS-24118
