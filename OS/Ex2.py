class Process:
    def __init__(self, pid, burst_time, priority=None):
        self.pid = pid
        self.burst_time = burst_time
        self.priority = priority
        self.waiting_time = 0
        self.turnaround_time = 0

def fcfs(processes):
    time = 0
    for process in processes:
        process.waiting_time = time
        process.turnaround_time = process.waiting_time + process.burst_time
        time += process.burst_time

def sjf(processes):
    processes.sort(key=lambda x: x.burst_time)
    fcfs(processes)

def priority_scheduling(processes):
    processes.sort(key=lambda x: x.priority)
    fcfs(processes)

def round_robin(processes, quantum):
    queue = processes[:]
    time = 0
    while queue:
        process = queue.pop(0)
        if process.burst_time > quantum:
            process.burst_time -= quantum
            time += quantum
            queue.append(process)
        else:
            time += process.burst_time
            process.waiting_time = time - process.burst_time
            process.turnaround_time = process.waiting_time + process.burst_time

def print_results(processes, algorithm_name):
    print(f"\n{algorithm_name} Scheduling:\n")
    print("PID\tBurst Time\tWaiting Time\tTurnaround Time")
    for p in processes:
        print(f"{p.pid}\t{p.burst_time}\t{p.waiting_time}\t{p.turnaround_time}")

# Example Usage
processes = [Process(1, 6, 2), Process(2, 8, 1), Process(3, 7, 3), Process(4, 3, 4)]
fcfs(processes)
print_results(processes, "FCFS")

sjf(processes)
print_results(processes, "SJF")

priority_scheduling(processes)
print_results(processes, "Priority")

round_robin(processes, quantum=3)
print_results(processes, "Round Robin")
