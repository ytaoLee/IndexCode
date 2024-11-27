class Index:
    def __init__(self, space, dim, max_elements, M=x, ef_construction=y, num_threads=z):
        """
        Initialize the index with multithreading support
        :param space: Distance metric ('l2' or 'ip')
        :param dim: Dimension of the vectors
        :param max_elements: Maximum number of elements in the index
        :param M: Maximum number of connections per node at each level
        :param ef_construction: Precision during index construction
        :param num_threads: Number of threads to use for parallel processing
        """
        self.space = space
        self.dim = dim
        self.max_elements = max_elements
        self.M = M
        self.ef_construction = ef_construction
        self.num_threads = num_threads  # Number of threads for parallel execution
        self.entry_point = None
        self.levels = int(math.log2(max_elements))
        self.index = np.zeros((max_elements, dim), dtype=np.float32)
        self.labels = np.zeros(max_elements, dtype=np.int32)
        self.connections = {level: [[] for _ in range(max_elements)] for level in range(self.levels + 1)}
        self.element_count = 0

    def _get_distance(self, a, b):
        """
        Calculate the distance between two vectors
        :param a: Vector a
        :param b: Vector b
        :return: Distance
        """
        if self.space == 'l2':
            return np.linalg.norm(a - b)
        elif self.space == 'ip':
            return -np.dot(a, b)  # Use negative for maximum inner product

    def _search_level(self, entry_point, query, ef, level):
        """
        Greedy search within a specific layer to find nearest neighbors
        :param entry_point: Starting node for search
        :param query: Query vector
        :param ef: Number of neighbors to find
        :param level: Current layer level
        :return: List of nearest neighbors
        """
        candidates = [(self._get_distance(query, self.index[entry_point]), entry_point)]
        visited = set([entry_point])
        nearest_neighbors = []

        while candidates:
            dist, current_node = heapq.heappop(candidates)

            if len(nearest_neighbors) < ef:
                heapq.heappush(nearest_neighbors, (-dist, current_node))
            elif -dist > nearest_neighbors[0][0]:
                heapq.heappushpop(nearest_neighbors, (-dist, current_node))

            for neighbor in self.connections[level][current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    neighbor_dist = self._get_distance(query, self.index[neighbor])
                    heapq.heappush(candidates, (neighbor_dist, neighbor))

        return [node for _, node in sorted(nearest_neighbors, reverse=True)]

    def _select_neighbors(self, candidates, M):
        """
        Select M nearest neighbors from candidates
        :param candidates: List of candidate neighbors
        :param M: Maximum number of neighbors to select
        :return: Selected neighbors
        """
        neighbors = []
        for dist, idx in sorted(candidates):
            if len(neighbors) < M:
                neighbors.append((dist, idx))
            else:
                break
        return [idx for _, idx in neighbors]

    def insert(self, data, label):
        """
        Insert a new data point into the index
        :param data: Data point to insert
        :param label: Label for the data point
        """
        idx = self.element_count
        self.index[idx] = data
        self.labels[idx] = label
        max_level = int(math.floor(-math.log(random.uniform(0, 1)) * self.M))
        self.element_count += 1

        if self.entry_point is None:
            self.entry_point = idx
            for level in range(max_level + 1):
                self.connections[level][idx] = []
            return

        # Multithreading to parallelize the insertion at each level
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_level = {}
            current_node = self.entry_point

            for level in range(self.levels, max_level, -1):
                future = executor.submit(self._search_layer, current_node, data, 1, level)
                future_to_level[future] = level

            for future in as_completed(future_to_level):
                level = future_to_level[future]
                neighbors = future.result()
                if neighbors:
                    current_node = neighbors[0]

            # Insert into levels from max_level to 0
            for level in range(max_level, -1, -1):
                neighbors = self._search_level(current_node, data, self.ef_construction, level)
                selected_neighbors = self._select_neighbors([(self._get_distance(data, self.index[n]), n) for n in neighbors], self.M)
                self.connections[level][idx] = selected_neighbors
                for neighbor in selected_neighbors:
                    self.connections[level][neighbor].append(idx)
                current_node = neighbors[0] if neighbors else current_node

    def knn_query(self, query, k):
        """
        Perform k-nearest neighbor search with multithreading
        :param query: Query vector
        :param k: Number of nearest neighbors
        :return: List of nearest neighbor labels and distances
        """
        current_node = self.entry_point

        # Multithreaded search across levels
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_level = {}
            for level in range(self.levels, -1, -1):
                future = executor.submit(self._search_layer, current_node, query, 1, level)
                future_to_level[future] = level

            for future in as_completed(future_to_level):
                level = future_to_level[future]
                neighbors = future.result()
                if neighbors:
                    current_node = neighbors[0]

        # Final search on the lowest layer to get k nearest neighbors
        nearest_neighbors = self._search_level(current_node, query, self.ef_construction, 0)

        top_k = []
        for node in nearest_neighbors:
            dist = self._get_distance(query, self.index[node])
            if len(top_k) < k:
                heapq.heappush(top_k, (-dist, self.labels[node]))
            else:
                if -dist > top_k[0][0]:
                    heapq.heappushpop(top_k, (-dist, self.labels[node]))

        return [label for _, label in sorted(top_k, reverse=True)]

    def set_ef(self, ef_search):
        """
        Set precision during query search
        :param ef_search: Precision during search
        """
        self.ef_construction = ef_search

    def save(self, filename):
        """
        Save the index to a file
        :param filename: File name
        """
        np.savez_compressed(filename, index=self.index, labels=self.labels)

    def load(self, filename):
        """
        Load the index from a file
        :param filename: File name
        """
        data = np.load(filename, allow_pickle=True)
        self.index = data['index']
        self.labels = data['labels']
