import heapq

class Process:
    def __init__(self, pid, arrival_time, burst_time, priority):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.priority = priority
        self.remaining_time = burst_time
        self.start_time = -1
        self.completion_time = -1
        self.waiting_time = 0
        self.turnaround_time = 0

def priority_scheduling(processes, preemptive=True):
    processes.sort(key=lambda x: (x.arrival_time, x.priority))
    ready_queue = []
    time = 0
    executed_processes = []
    
    while processes or ready_queue:
        while processes and processes[0].arrival_time <= time:
            heapq.heappush(ready_queue, (processes[0].priority, processes.pop(0)))
        
        if ready_queue:
            priority, current = heapq.heappop(ready_queue)
            if current.start_time == -1:
                current.start_time = time
            
            execution_time = 1 if preemptive else current.remaining_time
            current.remaining_time -= execution_time
            time += execution_time
            
            if current.remaining_time > 0:
                heapq.heappush(ready_queue, (current.priority, current))
            else:
                current.completion_time = time
                current.turnaround_time = current.completion_time - current.arrival_time
                current.waiting_time = current.turnaround_time - current.burst_time
                executed_processes.append(current)
        else:
            time += 1
    
    return executed_processes

def print_process_info(processes):
    print("PID\tAT\tBT\tPR\tCT\tTAT\tWT")
    for p in processes:
        print(f"{p.pid}\t{p.arrival_time}\t{p.burst_time}\t{p.priority}\t{p.completion_time}\t{p.turnaround_time}\t{p.waiting_time}")

if __name__ == "__main__":
    processes = [
        Process(1, 0, 8, 2),
        Process(2, 1, 4, 1),
        Process(3, 2, 9, 3),
        Process(4, 3, 5, 2)
    ]
    executed = priority_scheduling(processes, preemptive=True)
    print_process_info(executed)
