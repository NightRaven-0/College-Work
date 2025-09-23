import heapq
from collections import deque

def fcfs(processes):
    processes.sort(key=lambda x: x[1])  # Sort by arrival time
    time, schedule = 0, []
    for pid, arrival, burst in processes:
        time = max(time, arrival) + burst
        schedule.append((pid, time))
    return schedule

def sjf(processes):
    processes.sort(key=lambda x: x[1])  # Sort by arrival time
    pq, time, schedule = [], 0, []
    i, n = 0, len(processes)
    while i < n or pq:
        while i < n and processes[i][1] <= time:
            heapq.heappush(pq, (processes[i][2], processes[i][0]))
            i += 1
        if pq:
            burst, pid = heapq.heappop(pq)
            time += burst
            schedule.append((pid, time))
        else:
            time = processes[i][1] if i < n else time
    return schedule

def priority_scheduling(processes):
    processes.sort(key=lambda x: x[1])
    pq, time, schedule = [], 0, []
    i, n = 0, len(processes)
    while i < n or pq:
        while i < n and processes[i][1] <= time:
            heapq.heappush(pq, (processes[i][3], processes[i][2], processes[i][0]))
            i += 1
        if pq:
            priority, burst, pid = heapq.heappop(pq)
            time += burst
            schedule.append((pid, time))
        else:
            time = processes[i][1] if i < n else time
    return schedule

def round_robin(processes, quantum):
    processes.sort(key=lambda x: x[1])
    queue = deque(processes)
    time, schedule = 0, {}
    while queue:
        pid, arrival, burst = queue.popleft()
        if burst > quantum:
            time += quantum
            queue.append((pid, arrival, burst - quantum))
        else:
            time += burst
            schedule[pid] = time
    return schedule

# Example Usage
processes = [(1, 0, 5, 2), (2, 1, 3, 1), (3, 2, 8, 3), (4, 3, 6, 2)]
print("FCFS:", fcfs(processes))
print("SJF:", sjf(processes))
print("Priority Scheduling:", priority_scheduling(processes))
print("Round Robin:", round_robin(processes, quantum=2))
