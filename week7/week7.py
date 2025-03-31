import heapq
from datetime import datetime, timedelta

# Graph with (neighbor, base travel time)
graph = {
    'Hub': [('A', 10), ('B', 15)],
    'A': [('C', 10), ('D', 30)],
    'B': [('D', 5)],
    'C': [('D', 5), ('E', 15)],
    'D': [('E', 10)],
    'E': []
}

# Traffic delays 
real_time_traffic = {
    ('A', 'C'): 2,
    ('C', 'D'): 3,
    ('B', 'D'): 0,
    ('D', 'E'): 5,
}

# Food prep times in minutes
food_prep_time = {
    'C': 5,
    'D': 2,
    'E': 0
}

# Delivery time windows
time_windows = {
    'C': (15, 45),  # 15–45 mins window
    'D': (0, 40),
    'E': (20, 60)
}

# Stability threshold in minutes
stability_threshold = 2

def get_travel_time(u, v, base):
    return base + real_time_traffic.get((u, v), 0)

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    prev = {node: None for node in graph}
    distances[start] = 0
    queue = [(0, start)]

    while queue:
        curr_dist, curr_node = heapq.heappop(queue)
        if curr_dist > distances[curr_node]:
            continue

        for neighbor, base in graph[curr_node]:
            time = get_travel_time(curr_node, neighbor, base)
            new_dist = curr_dist + time
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                prev[neighbor] = curr_node
                heapq.heappush(queue, (new_dist, neighbor))

    return distances, prev

def is_within_time_window(dest, eta):
    if dest not in time_windows:
        return True
    start, end = time_windows[dest]
    return start <= eta <= end

def reconstruct_path(prev, end):
    path = []
    while end:
        path.insert(0, end)
        end = prev[end]
    return path

def multi_stop_route(start, stops):
    visited = set()
    current = start
    total_time = 0
    route = [start]
    delivery_times = {}

    while len(visited) < len(stops):
        distances, prev = dijkstra(graph, current)
        candidates = [(stop, distances[stop]) for stop in stops if stop not in visited and is_within_time_window(stop, total_time + distances[stop] + food_prep_time.get(stop, 0))]
        if not candidates:
            break
        next_stop, travel_time = min(candidates, key=lambda x: x[1])
        total_time += travel_time + food_prep_time.get(next_stop, 0)
        route += reconstruct_path(prev, next_stop)[1:]
        delivery_times[next_stop] = total_time
        visited.add(next_stop)
        current = next_stop

    return route, delivery_times

def is_significant_change(current_time, new_time):
    return abs(current_time - new_time) >= stability_threshold

class RouteOptimizer:
    def __init__(self):
        self.last_routes = {}

    def optimize(self, stops):
        route, times = multi_stop_route('Hub', stops)
        stable_output = {}
        for stop in stops:
            prev = self.last_routes.get(stop)
            new = times.get(stop, float('inf'))
            if prev is None or is_significant_change(prev, new):
                stable_output[stop] = new
                self.last_routes[stop] = new
            else:
                stable_output[stop] = prev
        return route, stable_output

if __name__ == '__main__':
    stops = ['C', 'D', 'E']
    optimizer = RouteOptimizer()
    route, deliveries = optimizer.optimize(stops)

    print("\nOptimized Multi-Stop Delivery Route:")
    print(" -> ".join(route))

    print("\nEstimated Delivery Times (mins):")
    for stop, t in deliveries.items():
        print(f"  {stop}: {t} mins")
