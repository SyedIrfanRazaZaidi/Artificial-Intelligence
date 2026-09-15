# Lab Session 06 - Iterative Deepening Search (IDS)
# Graph taken from the diagram. All path costs = 1, all heuristics = 0.

graph = {
    'S': ['D', 'A'],
    'A': ['B', 'C'],
    'D': ['B', 'E'],
    'B': ['E', 'C'],
    'C': ['G'],
    'E': ['G'],
    'G': []
}

cost = 1
heuristic = {
    'S': 0, 'A': 0, 'B': 0, 'C': 0,
    'D': 0, 'E': 0, 'G': 0
}


def depth_limited_search(path, goal, limit):
    node = path[-1]

    if node == goal:
        return path

    if limit <= 0:
        return None

    for neighbour in graph[node]:
        if neighbour not in path:          # avoid revisiting nodes in same path
            new_path = path + [neighbour]
            result = depth_limited_search(new_path, goal, limit - 1)
            if result is not None:
                return result

    return None


def ids(start, goal, max_depth):
    for limit in range(0, max_depth + 1):
        print("Trying depth limit = " + str(limit))
        result = depth_limited_search([start], goal, limit)
        if result is not None:
            return result, limit
    return None, None


def show(path, limit):
    if path is None:
        print("No path found")
        return

    total = (len(path) - 1) * cost

    line = ""
    for i in range(len(path)):
        line = line + path[i]
        if i != len(path) - 1:
            line = line + " -> "

    print("")
    print("Iterative Deepening Search (IDS)")
    print("  Depth limit reached at : " + str(limit))
    print("  Path        : " + line)
    print("  Path cost   : " + str(total))
    print("  f = g + h   : " + str(total) + " + " + str(heuristic[path[-1]]) +
          " = " + str(total + heuristic[path[-1]]))


start = 'S'
goal = 'G'
max_depth = 5

print("Start node: " + start + "    Goal node: " + goal)
print("")

path, limit = ids(start, goal, max_depth)
show(path, limit)
