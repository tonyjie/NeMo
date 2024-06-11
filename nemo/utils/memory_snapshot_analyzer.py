import pickle
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, peak_prominences
from nemo.utils import logging

__all__ = ['peak_memory_analysis']

GB_SIZE = 1024*1024*1024
MB_SIZE = 1024*1024
KB_SIZE = 1024

def load_pickle_file(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def make_hashable(obj):
    """
    Helper function to make an object hashable. 
    """
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    if isinstance(obj, list):
        return tuple(make_hashable(e) for e in obj)
    return obj


def read_tp(timepoint):
    time_us = timepoint['time_us']
    addr = timepoint['addr']
    action = timepoint['action']
    size = timepoint['size'] / GB_SIZE
    frames = timepoint['frames']
    stream = timepoint['stream']
    return (time_us, addr, action, size, frames, stream)


def first_py_frame(alloc_frames):
    """
    Return the frame when it first shows a `.py` file.
    """
    # target_string = 'layernorm_linear.py'
    for frame in alloc_frames:
        if '.py' in frame['filename']:
            return frame
    return None

def alloc_memory_timeline(one_trace):
    curr_alloc_memory = 0 # in GB

    min_alloc_memory = float('inf')
    max_alloc_memory = float('-inf')

    time_min_alloc = None
    time_max_alloc = None
    idx_min = None
    idx_max = None

    FILTER_FACTOR = 3 # only check the first (1/FILTER_FACTOR) of trace
    
    for idx, timepoint in enumerate(one_trace):
        if idx > len(one_trace) / FILTER_FACTOR:
            break

        (time_us, addr, action, size, frames, stream) = read_tp(timepoint)

        if (action == "alloc"):
            curr_alloc_memory += size
        elif (action == "free_completed"):
            curr_alloc_memory -= size

        if curr_alloc_memory > max_alloc_memory:
            max_alloc_memory = curr_alloc_memory
            time_max_alloc = time_us
            idx_max = idx

        if curr_alloc_memory < min_alloc_memory:
            min_alloc_memory = curr_alloc_memory
            time_min_alloc = time_us
            idx_min = idx
            
        # logging.info(f"curr_alloc_memory: {curr_alloc_memory} MB")
        # logging.info(f"max_alloc_memory: {max_alloc_memory} MB")
        # logging.info(f"min_alloc_memory: {min_alloc_memory} MB")

    return (idx_min, time_min_alloc, min_alloc_memory), (idx_max, time_max_alloc, max_alloc_memory)


def record_alloc_memory_timeline(trace):
    alloc_memory_history = np.zeros(len(trace)) # record the curr_alloc_memory at each timepoint
    time_us_history = np.zeros(len(trace)) # record the time_us at each timepoint
    
    curr_alloc_memory = 0 # in GB

    min_alloc_memory = float('inf')
    max_alloc_memory = float('-inf')
    
    time_min_alloc = None
    time_max_alloc = None
    idx_min = None
    idx_max = None
    
    for idx, timepoint in enumerate(trace):
        (time_us, addr, action, size, frames, stream) = read_tp(timepoint)

        if (action == "alloc"):
            curr_alloc_memory += size
        elif (action == "free_completed"):
            curr_alloc_memory -= size

        alloc_memory_history[idx] = curr_alloc_memory
        time_us_history[idx] = time_us
        
        
        if curr_alloc_memory > max_alloc_memory:
            max_alloc_memory = curr_alloc_memory
            time_max_alloc = time_us
            idx_max = idx

        if curr_alloc_memory < min_alloc_memory:
            min_alloc_memory = curr_alloc_memory
            time_min_alloc = time_us
            idx_min = idx
            
        # logging.info(f"curr_alloc_memory: {curr_alloc_memory} MB")
        # logging.info(f"max_alloc_memory: {max_alloc_memory} MB")
        # logging.info(f"min_alloc_memory: {min_alloc_memory} MB")

    return (alloc_memory_history, time_us_history), (idx_min, time_min_alloc, min_alloc_memory), (idx_max, time_max_alloc, max_alloc_memory)


class MemoryLife:
    """
    Records one active memory interval. Information includes its lifetime interval (start, end), the size of memory, and related frames (stack trace). 
    """
    def __init__(self, start_time_us, end_time_us, size, alloc_frames, free_frames):
        self.start_time_us = start_time_us
        self.end_time_us = end_time_us
        self.size = size
        self.alloc_frames = alloc_frames
        self.free_frames = free_frames
    
    # the default print function
    def __str__(self):
        return f"[{self.start_time_us}, {self.end_time_us}], size: {self.size:.5f} GB"
    
    

class MemoryInfo:
    def __init__(self, addr):
        self.addr = addr
        self.history = [] # list of tps
        self.lifetime = [] # list of MemoryLife

    def add_history(self, tp):
        self.history.append(tp)

    def add_lifetime_alloc(self, start_time_us, size, alloc_frames):
        # self.lifetime.append([time_us, None]) # start a new interval
        self.lifetime.append(MemoryLife(start_time_us, None, size, alloc_frames, None))

    def add_lifetime_free(self, end_time_us, size, frames):
        # Check exception
        # 1. if the last lifetime is already freed, then raise exception
        if self.lifetime[-1].end_time_us is not None:
            raise Exception(f"Memory at addr {self.addr} is already freed")
        # 2. check if the alloctaed size == freed size
        if self.lifetime[-1].size != size:
            raise Exception(f"Size mismatch (allocated size vs freed size): {self.lifetime[-1].size} vs {size}")
        # record the end time
        self.lifetime[-1].end_time_us = end_time_us
        # record the free frames
        self.lifetime[-1].free_frames = frames
    
    def get_info_if_alive(self, time_us):
        """
        Check if the memory is alive at the given time point. We all use relative time point. 
        Return (start_time_us, size, alloc_frames) if alive, otherwise return None.
        """
        for life in self.lifetime:
            if (life.start_time_us <= time_us) and (life.end_time_us is None or life.end_time_us >= time_us):
                return (life.start_time_us, life.size, life.alloc_frames)
        return None


    def print_info(self):
        logging.info(f"Addr: {self.addr}, len(lifetime): {len(self.lifetime)}")
        # print the first 5 lifetimes
        for idx, life in enumerate(self.lifetime):
            if idx < 5:
                logging.info(life)


class MemoryTracker:
    def __init__(self):
        self.data = {} # key: addr, value: MemoryInfo

    def add_entry(self, tp):
        (time_us, addr, action, size, frames, stream) = read_tp(tp)
        
        if addr not in self.data:
            self.data[addr] = MemoryInfo(addr)

        self.data[addr].add_history(tp)
        
        if action == 'alloc': 
            # alloc memory
            self.data[addr].add_lifetime_alloc(time_us, size, frames)
        elif (action == 'free_completed') and (len(self.data[addr].lifetime)!= 0) :
            # free memory, but this can't be the first memory op for this addr. There must be an alloc happening before. 
            self.data[addr].add_lifetime_free(time_us, size, frames)
        else:
            # ignore free_requested; or free_completed op before any alloc op, since that is related to some previous alloc ops that are not captured
            pass
    
    def check_alive_memory(self, time_us):
        """
        Gather all alive memory addresses, their start time, their sizes, and alloc_frames at the given time point. Save it with the order of start time. 
        """
        alive_memory = []
        for addr, memory_info in self.data.items():
            info = memory_info.get_info_if_alive(time_us) # info = (start_time_us, size, alloc_frames)
            if info is not None:
                alive_memory.append((addr, info[0], info[1], info[2]))
        alive_memory.sort(key=lambda x: x[1]) # sort by start time
        return alive_memory

'''
Data structure for memory grouped by alloc_frames. 
It includes alloc_frames, a list of memory that shares the same alloc_frames, the total count, the total size of memory. 
'''
class MemoryGroupByAllocFrames:
    def __init__(self, alloc_frames, memory_list):
        self.alloc_frames = alloc_frames
        self.memory_list = memory_list
        self.count = len(memory_list)
        self.total_size = sum([x[2] for x in memory_list])
        self.frame_string = self.first_py_frame()
    
    def add_memory(self, memory):
        self.memory_list.append(memory) # memory = (addr, start_time_us, size, alloc_frames)
        self.count += 1
        self.total_size += memory[2]
    
    def first_py_frame(self):
        """
        Return the frame when it first shows a `.py` file.
        """
        # target_string = 'layernorm_linear.py'
        for frame in self.alloc_frames:
            if '.py' in frame['filename']:
                return frame
        return None


    def __str__(self):
        # return f"Alloc Frames: {self.alloc_frames}, Count: {self.count}, Total Size: {self.total_size:.5f} GB"
        # return f"Count: {self.count}, Total Size: {self.total_size:.5f} GB"
        return f"First Py Frame: {self.frame_string}, Count: {self.count}, Total Size: {self.total_size:.5f} GB"


class MemoryAnalyzer:
    def __init__(self, memory_list):
        """
        memory_list: list of (addr, start_time_us, size, alloc_frames)
        """
        self.memory_list = memory_list
        self.alloc_frames_group = dict() # key: alloc_frames, value: MemoryGroupByAllocFrames

    def group_memory_by_alloc_frames(self):
        for addr, start_time_us, size, alloc_frames in self.memory_list:
            alloc_frames_tuple = make_hashable(alloc_frames) # make it hashable so that it can be used as a key
            
            if alloc_frames_tuple not in self.alloc_frames_group:
                self.alloc_frames_group[alloc_frames_tuple] = MemoryGroupByAllocFrames(alloc_frames, [])

            self.alloc_frames_group[alloc_frames_tuple].add_memory((addr, start_time_us, size, alloc_frames))
    
    def print_info(self):
        for frames, group in self.alloc_frames_group.items():
            logging.info(f"{group}")
    
    def save_as_list(self):
        """
        Save as a list of tuple (frame_string, count, total_size)
        """
        memory_group_by_alloc_frames = []
        for frames, group in self.alloc_frames_group.items():
            one_layer = group.frame_string, group.count, group.total_size
            memory_group_by_alloc_frames.append(one_layer)
        return memory_group_by_alloc_frames


def find_prominence_peak(history, prominence):
    """
    Find peaks/valleys with prominence. 
    After checking the definition of prominence, seems like in our case, prominence is close to the memory_gap. So we use a 0.7*memory_gap as prominence. 
    Note that this is kind of heuristic, and can be broken in some cases (?)

    Return: prominence_peaks and prominence_valleys
        (peaks, valleys)
    """
    peaks, properties_peak = find_peaks(x=history, prominence=prominence)  # looks like prominence is close to max_alloc_memory
    # Find local minima (inverted peaks)
    inverted_history = -history
    valleys, properties_valleys = find_peaks(inverted_history, prominence=prominence)
    return (peaks, properties_peak), (valleys, properties_valleys)

def find_paired_peak_valley(peaks, valleys):
    """
    Ensure each trough is paired with the next peak
    """
    paired_peaks = []
    paired_valleys = []

    for valley in valleys:
        # Find the next peak after this trough
        next_peak_candidates = peaks[peaks > valley]
        if next_peak_candidates.size > 0:
            next_peak = next_peak_candidates[0]
            paired_valleys.append(valley)
            paired_peaks.append(next_peak)
    return paired_peaks, paired_valleys

def to_relative_time(trace, TIME_OFFSET):
    """
    Change the absolute time to the relative time for all the timepoints in this trace. The absolute time is too large, which is not readable. 
    """
    for timepoint in trace:
        timepoint['time_us'] -= TIME_OFFSET
    return trace


def peak_memory_analysis(mem_snapshot_filepath, mem_snapshot_csv_dir):
    # ===== Loading =====
    snapshot = load_pickle_file(mem_snapshot_filepath)
    traces = snapshot['device_traces']
    trace = traces[0] # useful trace. Device 0. 

    min_time_us, max_time_us = trace[0]['time_us'], trace[-1]['time_us']
    TIME_OFFSET = min_time_us
    min_time_us -= TIME_OFFSET
    max_time_us -= TIME_OFFSET

    # change all the global time_us to the relative time (starting from 0)
    trace = to_relative_time(trace, TIME_OFFSET)


    ##### Here the only thing we need to update is: instead of return (idx_min, idx_max). We want to print all the local min/max idx. 
    (alloc_memory_history, time_us_history), (idx_min, time_min_alloc, min_alloc_memory), (idx_max, time_max_alloc, max_alloc_memory) = record_alloc_memory_timeline(trace)

    memory_gap_GB = (max_alloc_memory - min_alloc_memory)
    # logging.info(f"Max: {max_alloc_memory:.5f} GB , Min: {min_alloc_memory:.5f} GB, Gap: {memory_gap_GB:.5f} GB")

    # Find peak (local max) & troughs (local min)
    prominence = 0.7 * memory_gap_GB
    (peaks, properties_peak), (valleys, properties_valleys) = find_prominence_peak(alloc_memory_history, prominence)
    
    # Check the length of peaks, valleys and its prominence
    logging.info(f"===== Check Peaks and Valleys =====")
    logging.info(f"Number of peaks: {len(peaks)}, Height: {alloc_memory_history[peaks]}, Peak Prominence: {properties_peak['prominences']}") # the prominence of each peak
    logging.info(f"Number of valleys: {len(valleys)}, Height: {alloc_memory_history[valleys]}, Valley Prominence: {properties_valleys['prominences']}") # the prominence of each valley

    # Ensure each valley is paired with the next peak. 
    paired_peaks, paired_valleys = find_paired_peak_valley(peaks, valleys)

    # From the first valley to the first peak
    trace_first_rise = trace[paired_valleys[0] : paired_peaks[0] + 1]
    time_first_valley = time_us_history[paired_valleys[0]]
    time_first_peak = time_us_history[paired_peaks[0]]


    # track the lifetime of each memory address
    tracker = MemoryTracker()
    for tp in trace:
        tracker.add_entry(tp)
    

    # check the first 10 memory addresses
    # for idx, (k, v) in enumerate(tracker.data.items()):
    #     if idx < 5:
    #         v.print_info()
    
    # Now I want to check the status given a specific time point. I want to know which memory addr is alive at the time point, and what is the size of each memory addr.
    logging.info(f"===== Check Alive Memory =====")
    logging.info(f"Check the status at time_first_valley")
    alive_memory_min = tracker.check_alive_memory(time_first_valley)
    # the accumulated memory size at time_min_alloc
    total_memory_min = sum([x[2] for x in alive_memory_min])
    logging.info(f"Number of alive memory addresses: {len(alive_memory_min)}, Total memory size: {total_memory_min:.5f} GB")



    logging.info(f"Check the status at time_first_peak") # need to be the time here
    alive_memory_max = tracker.check_alive_memory(time_first_peak)
    # the accumulated memory size at time_max_alloc
    total_memory_max = sum([x[2] for x in alive_memory_max])
    logging.info(f"Number of alive memory addresses: {len(alive_memory_max)}, Total memory size: {total_memory_max:.5f} GB")

    logging.info(f"Memory Gap between a valley and a peak: {total_memory_max - total_memory_min:.5f} GB")


    logging.info(f"===== Export to CSV =====")
    # 1. export this alive_memory_max to a csv file, with pandas
    alive_memory_max_short = [(x[0], x[1], x[2], first_py_frame(x[3])) for x in alive_memory_max]
    data = {
        "Addr (Decimal)": [x[0] for x in alive_memory_max],
        "Time of Allocation (us)": [x[1] for x in alive_memory_max],
        "Size (GB)": [x[2] for x in alive_memory_max],
        "First Python Frame": [x[3] for x in alive_memory_max_short], # this is the first frame that shows a `.py` file
        "Full Stack Trace": [x[3] for x in alive_memory_max]
    }
    df = pd.DataFrame(data)
    csv_1_path = os.path.join(mem_snapshot_csv_dir, "alive_memory_peak_full_stack_trace.csv")
    df.to_csv(csv_1_path, index=False) # this generates a big csv file, since the alloc_frames are huge. 
    logging.info(f"1. Exported to {csv_1_path}")

    # 2. group memory by alloc_frames
    analyzer = MemoryAnalyzer(alive_memory_max)
    analyzer.group_memory_by_alloc_frames()
    # analyzer.print_info()
    memory_group_by_alloc_frames = analyzer.save_as_list()


    # logging.info(memory_group_by_alloc_frames)
    # export as csv
    data_group = {
        "First Python Frame": [x[0] for x in memory_group_by_alloc_frames],
        "Repeat": [x[1] for x in memory_group_by_alloc_frames],
        "Total Size (GB)": [x[2] for x in memory_group_by_alloc_frames]
    }
    df_group = pd.DataFrame(data_group)
    csv_2_path = os.path.join(mem_snapshot_csv_dir, "memory_group_by_alloc_frames.csv")
    df_group.to_csv(csv_2_path, index=False)
    logging.info(f"2. Exported to {csv_2_path}")