# Lab Session 06 - BFS and DFS
# Graph taken from the diagram. All path costs = 1, all heuristics = 0.

# adjacency list (directed graph)
graph = {
    'S': ['D', 'A'],
    'A': ['B', 'C'],
    'D': ['B', 'E'],
    'B': ['E', 'C'],
    'C': ['G'],
    'E': ['G'],
    'G': []
}

# every edge costs 1
cost = 1

# heuristic values (all zero as required)
heuristic = {
    'S': 0, 'A': 0, 'B': 0, 'C': 0,
    'D': 0, 'E': 0, 'G': 0
}


def bfs(start, goal):
    # queue holds paths, first path is just the start node
    queue = [[start]]
    visited = []

    while len(queue) > 0:
        path = queue[0]          # take the front path
        queue = queue[1:]        # remove it from the queue
        node = path[-1]          # last node of that path

        if node == goal:
            return path

        if node not in visited:
            visited.append(node)
            for neighbour in graph[node]:
                if neighbour not in visited:
                    new_path = path + [neighbour]
                    queue.append(new_path)   # add at the end -> FIFO

    return None


def dfs(start, goal):
    # stack holds paths, we take from the end -> LIFO
    stack = [[start]]
    visited = []

    while len(stack) > 0:
        path = stack[-1]         # take the last path
        stack = stack[:-1]       # remove it from the stack
        node = path[-1]

        if node == goal:
            return path

        if node not in visited:
            visited.append(node)
            children = graph[node]
            # reverse order so the left-most child is expanded first
            i = len(children) - 1
            while i >= 0:
                neighbour = children[i]
                if neighbour not in visited:
                    new_path = path + [neighbour]
                    stack.append(new_path)
                i = i - 1

    return None


def show(name, path):
    if path is None:
        print(name + ": no path found")
        return

    total = (len(path) - 1) * cost

    line = ""
    for i in range(len(path)):
        line = line + path[i]
        if i != len(path) - 1:
            line = line + " -> "

    print(name)
    print("  Path      : " + line)
    print("  Path cost : " + str(total))
    print("  f = g + h : " + str(total) + " + " + str(heuristic[path[-1]]) +
          " = " + str(total + heuristic[path[-1]]))
    print("")


start = 'S'
goal = 'G'

print("Start node: " + start + "    Goal node: " + goal)
print("")

show("BFS (Breadth First Search)", bfs(start, goal))
show("DFS (Depth First Search)", dfs(start, goal))
