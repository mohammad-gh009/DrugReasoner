from collections import defaultdict, deque


class KnowledgeGraph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, src, dest):
        self.graph[src].append(dest)
        self.graph[dest].append(src)

    def get_all_paths(
        self, start: str, end: str, max_length: int = 4, max_paths: int = 5
    ) -> list:
        """
        Find all paths between start and end nodes within constraints

        Args:
            start: Starting node
            end: Target node
            max_length: Maximum path length
            max_paths: Maximum number of paths to return

        Returns:
            List of paths, where each path is a list of nodes
        """

        def dfs(current, target, path, paths, visited):
            if len(paths) >= max_paths:
                return
            if len(path) > max_length:
                return
            if current == target:
                paths.append(path[:])
                return

            for neighbor in self.graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dfs(neighbor, target, path + [neighbor], paths, visited)
                    visited.remove(neighbor)

        paths: list[list[str]] = []
        visited: set[str] = {start}
        dfs(start, end, [start], paths, visited)
        return paths

    def shortest_path(self, start, end):
        if start == end:
            return {"distance": 0, "path": [start]}
        visited = set()
        queue = deque([(start, 0)])
        parent = {}

        while queue:
            node, distance = queue.popleft()

            if node == end:
                path = []
                current = end
                while current is not None:
                    path.append(current)
                    current = parent.get(current)
                path.reverse()
                return {"distance": distance, "path": path}

            if node not in visited:
                visited.add(node)

                for neighbor in self.graph[node]:
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))
                        parent[neighbor] = node

        return {"distance": -1, "path": None}

    def get_neighbors(self, node):
        return self.graph[node]

    def get_all_nodes(self):
        return list(self.graph.keys())
