import pickle
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, peak_prominences
from nemo.utils import logging

__all__ = ['peak_memory_analysis_weight', 'peak_memory_analysis_activation']

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

def alloc_memory_timeline(trace):
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


    return (alloc_memory_history, time_us_history), (idx_min, time_min_alloc, min_alloc_memory), (idx_max, time_max_alloc, max_alloc_memory)



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
            if (life.start_time_us <= time_us) and (life.end_time_us is None or life.end_time_us > time_us):
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
            # ignore free_requested; or free_completed op before any alloc op, since that is related to some previous alloc ops that are not captured. 
            # one thing need to keep in mind: we ignore `free_completed` memory that we don't see `alloc` before in this trace. However, these memory activities exist because our trace is not guaranteed to be complete. This may lead to some memory counting mismatch. 
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
        # return f"Alloc Frames: {self.alloc_frames}, Count: {self.count}, Total Size: {self.total_size:.5f} GB"a
        # return f"Count: {self.count}, Total Size: {self.total_size:.5f} GB"
        return f"First Py Frame: {self.frame_string}, Count: {self.count}, Total Size: {self.total_size:.5f} GB"


class MemoryAnalyzer:
    def __init__(self, memory_list):
        """
        memory_list: list of (addr, start_time_us, size, alloc_frames)
        """
        self.memory_list = memory_list
        self.alloc_frames_group = dict() # key: alloc_frames_tuple, value: MemoryGroupByAllocFrames

    def group_memory_by_alloc_frames(self):
        for addr, start_time_us, size, alloc_frames in self.memory_list:
            alloc_frames_tuple = make_hashable(alloc_frames) # make it hashable so that it can be used as a key
            
            if alloc_frames_tuple not in self.alloc_frames_group:
                self.alloc_frames_group[alloc_frames_tuple] = MemoryGroupByAllocFrames(alloc_frames, [])

            self.alloc_frames_group[alloc_frames_tuple].add_memory((addr, start_time_us, size, alloc_frames))
        
        # sort the alloc_frames_group by total_size
        self.alloc_frames_group = dict(sorted(self.alloc_frames_group.items(), key=lambda x: x[1].total_size, reverse=True))
    
    def print_info(self):
        for frames, group in self.alloc_frames_group.items():
            logging.info(f"{group}")
    
    def save_as_list(self):
        """
        Save as a list of tuple (frame_string, count, total_size, alloc_frames)
        """
        memory_group_by_alloc_frames = []
        for frames, group in self.alloc_frames_group.items():
            one_layer = group.frame_string, group.count, group.total_size, group.alloc_frames
            memory_group_by_alloc_frames.append(one_layer)
        return memory_group_by_alloc_frames


def find_prominence_peak(history, prominence, distance=1000):
    """
    Find peaks/valleys with prominence. 
    After checking the definition of prominence, seems like in our case, prominence is close to the memory_gap. So we use a 0.8*memory_gap as prominence. 
    Note that this is kind of heuristic, and can be broken in some cases (?)

    Return: prominence_peaks and prominence_valleys
        (peaks, valleys)
    """
    peaks, properties_peak = find_peaks(x=history, prominence=prominence, distance=distance)  # looks like prominence is close to memory_gap   ## we set a very small distance, since we find a case that two peaks are very close to each other.
    # Find local minima (inverted peaks)
    inverted_history = -history
    valleys, properties_valleys = find_peaks(inverted_history, prominence=prominence, distance=distance)
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


# ===== Function: for two time points, check the corresponding alive memory, and compare them to see: what's new, what's gone, what's unchanged.
def compare_alive_memory(tracker, time_us_1, time_us_2):
    alive_memory_1 = tracker.check_alive_memory(time_us_1)
    alive_memory_2 = tracker.check_alive_memory(time_us_2)

    # 1. what's new
    new_memory = [x for x in alive_memory_2 if x not in alive_memory_1]
    # 2. what's gone
    gone_memory = [x for x in alive_memory_1 if x not in alive_memory_2]
    # 3. what's unchanged
    unchanged_memory = [x for x in alive_memory_2 if x in alive_memory_1]
    # 4. new_memory and gone_memory could overlap a lot, but they only differ on the addr or start_time_us. We compare them to show real_new_memory and real_gone_memory. 
    # This time when we compare `in`, we compare (size, alloc_frames) instead of (addr, start_time_us, size, alloc_frames)
    real_new_memory = [x for x in new_memory if (x[2], x[3]) not in [(y[2], y[3]) for y in gone_memory]]
    real_gone_memory = [x for x in gone_memory if (x[2], x[3]) not in [(y[2], y[3]) for y in new_memory]]

    return (new_memory, gone_memory), (real_new_memory, real_gone_memory), unchanged_memory

def print_memory_list_summary(memory_list, name):
    total_size = sum([x[2] for x in memory_list])
    print(f"len({name}): {len(memory_list)}, Total memory size: {total_size:.5f} GB")

def prune_frames(frames):
    """
    The stack trace is too long, and include many not-so-useful frame. We prune those frames with '??' in the filename. 
    """
    return [frame for frame in frames if '??' not in frame['filename']]







def peak_memory_analysis_weight(mem_snapshot_filepath, mem_snapshot_csv_dir):
    """
    Key and Entry Function for memory analysis: Weight Memory
    """
    
    # ===== Loading =====
    snapshot = load_pickle_file(mem_snapshot_filepath)
    traces = snapshot['device_traces']
    trace = traces[0] # useful trace. Device 0. 

    # remove the last timepoint, if it is an oom action
    if trace[-1]['action'] == 'oom':
        trace = trace[:-1]

    min_time_us, max_time_us = trace[0]['time_us'], trace[-1]['time_us']
    TIME_OFFSET = min_time_us
    min_time_us -= TIME_OFFSET
    max_time_us -= TIME_OFFSET

    # change all the global time_us to the relative time (starting from 0)
    trace = to_relative_time(trace, TIME_OFFSET)

    (alloc_memory_history, time_us_history), (idx_min, time_min_alloc, min_alloc_memory), (idx_max, time_max_alloc, max_alloc_memory) = alloc_memory_timeline(trace)
    logging.info(f"===== Global Max and Min Memory =====")
    logging.info(f"idx_min: {idx_min}, relative_idx_min: {idx_min/len(trace):.5f}, time_min_alloc: {time_min_alloc}, min_alloc_memory: {min_alloc_memory:.5f} GB")
    logging.info(f"idx_max: {idx_max}, relative_idx_max: {idx_max/len(trace):.5f}, time_max_alloc: {time_max_alloc}, max_alloc_memory: {max_alloc_memory:.5f} GB")

    # track the lifetime of each memory address
    tracker = MemoryTracker()
    for tp in trace:
        tracker.add_entry(tp)
    
    # ======== 1. Global Peak Alive Memory Analsysis ========
    logging.info(f"\n======== 1. Global Peak Alive Memory Analysis (Weight Only) ========")
    logging.info(f"===== Check Alive Memory at Global Peak: time_max_alloc (relative time): {time_max_alloc/max_time_us:.5f} =====") # need to be the time here

    alive_memory_max = tracker.check_alive_memory(time_max_alloc)
    # the accumulated memory size at time_max_alloc
    total_memory_max = sum([x[2] for x in alive_memory_max])
    logging.info(f"Number of alive memory addresses: {len(alive_memory_max)}, Total memory size: {total_memory_max:.5f} GB")
    
    logging.info(f"===== Export to CSV: alive memory with its stack traces =====")
    # 1. export this alive_memory_max to a csv file, with pandas
    alive_memory_max_short = [(x[0], x[1], x[2], first_py_frame(x[3])) for x in alive_memory_max]
    data = {
        "Addr (Decimal)": [x[0] for x in alive_memory_max],
        "Time of Allocation (us)": [x[1] for x in alive_memory_max],
        "Size (GB)": [x[2] for x in alive_memory_max],
        "First Python Frame": [x[3] for x in alive_memory_max_short], # this is the first frame that shows a `.py` file
        "Full Stack Trace": [x[3] for x in alive_memory_max], 
        "Pruned Stack Trace": [prune_frames(x[3]) for x in alive_memory_max], # prune the stack trace
    }
    df = pd.DataFrame(data)
    csv_1_path = os.path.join(mem_snapshot_csv_dir, "weight_alive_memory.csv")
    df.to_csv(csv_1_path, index=False) # this generates a big csv file, since the alloc_frames are huge. 
    logging.info(f"1: Exported to {csv_1_path}")


    # ======== 2. group memory by alloc_frames at global peak ========
    logging.info(f"\n======== 2. Group Global Peak Alive Memory by Alloc Frames ========")
    analyzer = MemoryAnalyzer(alive_memory_max)
    analyzer.group_memory_by_alloc_frames()
    # analyzer.print_info()
    memory_group_by_alloc_frames = analyzer.save_as_list()

    # logging.info(memory_group_by_alloc_frames)
    # export as csv
    logging.info(f"===== Export to CSV: grouped memory by alloc_frames =====")
    data_group = {
        "First Python Frame": [x[0] for x in memory_group_by_alloc_frames],
        "Repeat": [x[1] for x in memory_group_by_alloc_frames],
        "Total Size (GB)": [x[2] for x in memory_group_by_alloc_frames], 
        "Pruned Stack Trace": [prune_frames(x[3]) for x in memory_group_by_alloc_frames] # prune the stack trace
    }
    df_group = pd.DataFrame(data_group)
    csv_2_path = os.path.join(mem_snapshot_csv_dir, "weight_group_by_alloc_frames.csv")
    df_group.to_csv(csv_2_path, index=False)
    logging.info(f"2: Exported to {csv_2_path}")    





def peak_memory_analysis_activation(mem_snapshot_filepath, mem_snapshot_csv_dir, memory_allocated, num_mb):
    """
    Key and Entry Function for memory analysis: Activation Memory
    """
    memory_allocated_gb = memory_allocated / GB_SIZE
    NUM_MB = num_mb
    
    # ===== Loading =====
    snapshot = load_pickle_file(mem_snapshot_filepath)
    traces = snapshot['device_traces']
    trace = traces[0] # useful trace. Device 0. 

    # remove the last timepoint, if it is an oom action
    if trace[-1]['action'] == 'oom':
        trace = trace[:-1]

    min_time_us, max_time_us = trace[0]['time_us'], trace[-1]['time_us']
    TIME_OFFSET = min_time_us
    min_time_us -= TIME_OFFSET
    max_time_us -= TIME_OFFSET

    # change all the global time_us to the relative time (starting from 0)
    trace = to_relative_time(trace, TIME_OFFSET)


    (alloc_memory_history, time_us_history), (idx_min, time_min_alloc, min_alloc_memory), (idx_max, time_max_alloc, max_alloc_memory) = alloc_memory_timeline(trace)

    logging.info(f"======== 0. Peak and Valley Finding ========")
    logging.info(f"===== Global Max and Min Memory =====")
    logging.info(f"idx_min: {idx_min}, relative_idx_min: {idx_min/len(trace):.5f}, time_min_alloc: {time_min_alloc}, min_alloc_memory: {min_alloc_memory:.5f} GB")
    logging.info(f"idx_max: {idx_max}, relative_idx_max: {idx_max/len(trace):.5f}, time_max_alloc: {time_max_alloc}, max_alloc_memory: {max_alloc_memory:.5f} GB")


    # # [TODO] this memory_gap is not activation memory gap! min_alloc_memory will always be 0, if we start recording from the initialization. 
    memory_activation = max_alloc_memory - min_alloc_memory # Should be activation memory
    memory_max_total = max_alloc_memory + memory_allocated_gb
    memory_weight = memory_max_total - memory_activation # Should be weight memory (is there any other stationery memory?). But note that we don't count each weight buffer here. So that's just an approximation. 
    logging.info(f"Max Total: {memory_max_total:.5f} GB , Weight Memory: {memory_weight:.5f} GB, Activation Memory: {memory_activation:.5f} GB")


    # # Find peak (local max) & troughs (local min)
    # ========= Heuristics part: set prominence and distance =========
    prominence = 0.8 * memory_activation # this factor is kind of heuristic.
    distance = 1000 # min distance (index) between peaks. 
    (peaks, properties_peak), (valleys, properties_valleys) = find_prominence_peak(alloc_memory_history, prominence, distance=distance)
    
    # # Check the length of peaks, valleys and its prominence
    logging.info(f"===== Check Peaks and Valleys =====")
    logging.info(f"Number of peaks: {len(peaks)}")
    # peak_index/len(trace), shown in `.5f` format
    logging.info(f"Peak Index (Relative): {[f'{p/len(trace):.5f}' for p in peaks]}")
    # Height of each peak, shown in `.5f` format
    logging.info(f"Height of each peak: {[f'{alloc_memory_history[p]:.5f}' for p in peaks]}")
    # Prominence of each peak, shown in `.5f` format
    # logging.info(f"Prominence of each peak: {[f'{properties_peak['prominences'][i]:.5f}' for i in range(len(peaks))]}")

    logging.info(f"Number of valleys: {len(valleys)}")
    # valley_index/len(trace), shown in `.5f` format
    logging.info(f"Valley Index (Relative): {[f'{v/len(trace):.5f}' for v in valleys]}")
    # Height of each valley, shown in `.5f` format
    logging.info(f"Height of each valley: {[f'{alloc_memory_history[v]:.5f}' for v in valleys]}")
    # Prominence of each valley, shown in `.5f` format
    # logging.info(f"Prominence of each valley: {[f'{properties_valleys['prominences'][i]:.5f}' for i in range(len(valleys))]}")

    if (len(peaks) != 2 * NUM_MB):
        # raise Warning
        logging.info(f"***** Warning: The number of peaks is not 2*NUM_MB = {2*NUM_MB}. The number of peaks is {len(peaks)} *****")


    # # Ensure each valley is paired with the next peak. Pairs: [valley, peak]
    paired_peaks, paired_valleys = find_paired_peak_valley(peaks, valleys)

    # # handle the OOM case (if there is no up and down)
    if len(paired_peaks) == 0:
        # let paired_peaks be the last idx
        paired_peaks = [len(alloc_memory_history) - 1]
    if len(paired_valleys) == 0:
        # let paired_valleys be the first idx
        paired_valleys = [0]

    # print out the paired peaks and valleys
    logging.info(f"===== Check Paired Peaks and Valleys =====")
    logging.info(f"Number of paired peaks: {len(paired_peaks)}")
    logging.info(f"Number of paired valleys: {len(paired_valleys)}")
    # print paired peaks index and relative index
    logging.info(f"Paired Peaks: {paired_peaks}")
    logging.info(f"Relative Index of Paired Peaks: {[f'{p/len(trace):.5f}' for p in paired_peaks]}")
    logging.info(f"Paired Valleys: {paired_valleys}")
    logging.info(f"Relative Index of Paired Valleys: {[f'{v/len(trace):.5f}' for v in paired_valleys]}")

    logging.info(f"===== Check the closest local peak to the global peak =====")
    # check which peak this is, by comparing the idx_max with `peaks`. Select the closest one, and know this is n_th of the peaks. 
    # A little bit confused on these variables. `idx_peak` records the index of the original trace; `local_peak_idx` records the index of the peaks array, denoting which peak this is. 
    min_idx_diff = float('inf')
    for idx, idx_peak in enumerate(peaks):
        idx_diff = abs(idx_peak - idx_max)
        if idx_diff < min_idx_diff:
            min_idx_diff = idx_diff
            local_peak_idx = idx
    # Also check the difference is a reasonable small value: `min_idx_diff`
    logging.info(f"The closest local peak we find (to the global peak with our peak-finding algorithm) is the {local_peak_idx}-th peak, with a difference of {min_idx_diff} index. ")




    # # From the first valley to the first peak
    # trace_first_rise = trace[paired_valleys[0] : paired_peaks[0] + 1]
    # time_first_valley = time_us_history[paired_valleys[0]]
    # time_first_peak = time_us_history[paired_peaks[0]]


    # track the lifetime of each memory address
    tracker = MemoryTracker()
    for tp in trace:
        tracker.add_entry(tp)
    

    # check the first 10 memory addresses
    # logging.info(f"===== Check Memory Addresses Lifetime and Size =====")
    # for idx, (k, v) in enumerate(tracker.data.items()):
    #     if idx < 5:
    #         v.print_info()
    



    # ======== 1. Global Peak Alive Memory Analsysis ========

    # Now I want to check the status given a specific time point. I want to know which memory addr is alive at the time point, and what is the size of each memory addr.
    # logging.info(f"===== Check Alive Memory =====")
    # logging.info(f"Check the status at time_min_alloc: {time_min_alloc/max_time_us:.5f}")
    # alive_memory_min = tracker.check_alive_memory(time_min_alloc)
    # # the accumulated memory size at time_min_alloc
    # total_memory_min = sum([x[2] for x in alive_memory_min])
    # logging.info(f"Number of alive memory addresses: {len(alive_memory_min)}, Total memory size: {total_memory_min:.5f} GB")
    # logging.info(f"current memory: {alloc_memory_history[idx_min]:.5f} GB")

    logging.info(f"\n======== 1. Global Peak Alive Memory Analysis (Activation Only) ========")
    logging.info(f"===== Check Alive Memory at Global Peak: time_max_alloc (relative time): {time_max_alloc/max_time_us:.5f} =====") # need to be the time here

    


    alive_memory_max = tracker.check_alive_memory(time_max_alloc)
    # the accumulated memory size at time_max_alloc
    total_memory_max = sum([x[2] for x in alive_memory_max])
    logging.info(f"Number of alive memory addresses: {len(alive_memory_max)}, Total memory size: {total_memory_max:.5f} GB")
    
    logging.info(f"===== Export to CSV: alive memory with its stack traces =====")
    # 1. export this alive_memory_max to a csv file, with pandas
    alive_memory_max_short = [(x[0], x[1], x[2], first_py_frame(x[3])) for x in alive_memory_max]
    data = {
        "Addr (Decimal)": [x[0] for x in alive_memory_max],
        "Time of Allocation (us)": [x[1] for x in alive_memory_max],
        "Size (GB)": [x[2] for x in alive_memory_max],
        "First Python Frame": [x[3] for x in alive_memory_max_short], # this is the first frame that shows a `.py` file
        "Full Stack Trace": [x[3] for x in alive_memory_max], 
        "Pruned Stack Trace": [prune_frames(x[3]) for x in alive_memory_max], # prune the stack trace
    }
    df = pd.DataFrame(data)
    csv_1_path = os.path.join(mem_snapshot_csv_dir, "activation_alive_memory.csv")
    df.to_csv(csv_1_path, index=False) # this generates a big csv file, since the alloc_frames are huge. 
    logging.info(f"1: Exported to {csv_1_path}")


    # ======== 2. group memory by alloc_frames at global peak ========
    logging.info(f"\n======== 2. Group Global Peak Alive Memory by Alloc Frames ========")
    analyzer = MemoryAnalyzer(alive_memory_max)
    analyzer.group_memory_by_alloc_frames()
    # analyzer.print_info()
    memory_group_by_alloc_frames = analyzer.save_as_list()

    # logging.info(memory_group_by_alloc_frames)
    # export as csv
    logging.info(f"===== Export to CSV: grouped memory by alloc_frames =====")
    data_group = {
        "First Python Frame": [x[0] for x in memory_group_by_alloc_frames],
        "Repeat": [x[1] for x in memory_group_by_alloc_frames],
        "Total Size (GB)": [x[2] for x in memory_group_by_alloc_frames], 
        "Pruned Stack Trace": [prune_frames(x[3]) for x in memory_group_by_alloc_frames] # prune the stack trace
    }
    df_group = pd.DataFrame(data_group)
    csv_2_path = os.path.join(mem_snapshot_csv_dir, "activation_group_by_alloc_frames.csv")
    df_group.to_csv(csv_2_path, index=False)
    logging.info(f"2: Exported to {csv_2_path}")


    # ======== 3. Inter-iteration Comparison Analysis ========
    logging.info(f"\n======== 3. Inter-iteration Comparison Analysis ========")

    logging.info(f"===== Compare Alive Memory Between Each Consecutive Peaks =====")

    # do this in a loop, by comparing the memory between paired_peaks[i] and paired_peaks[i+1]
    # for i in range(len(paired_peaks) - 1):
    for i in range(len(peaks) - 1):
        time_peak_i = time_us_history[peaks[i]]
        time_peak_i_plus_1 = time_us_history[peaks[i+1]]
        (new_memory, gone_memory), (real_new_memory, real_gone_memory), unchanged_memory = compare_alive_memory(tracker, time_peak_i, time_peak_i_plus_1)
        logging.info(f"===== Compare Alive Memory between peaks[{i}] and peaks[{i+1}] =====")
        # print the curr_memory at time_peak_i and time_peak_i_plus_1
        logging.info(f"Memory at peak[{i}]: {alloc_memory_history[peaks[i]]:.5f}; Memory at peak[{i+1}]: {alloc_memory_history[peaks[i+1]]:.5f}; Delta: {alloc_memory_history[peaks[i+1]] - alloc_memory_history[peaks[i]]:.5f}")

        # print_memory_list_summary(new_memory, "new_memory")
        # print_memory_list_summary(gone_memory, "gone_memory")
        # print_memory_list_summary(unchanged_memory, "unchanged_memory")
        print_memory_list_summary(real_new_memory, "new_memory")
        print_memory_list_summary(real_gone_memory, "gone_memory")

        # export real_new_memory and real_gone_memory to csv
        logging.info(f"===== Export to CSV: new_memory and gone_memory between peak[{i}] and peak[{i+1}] =====")
        data_new = {
            "Addr (Decimal)": [x[0] for x in real_new_memory],
            "Time of Allocation (us)": [x[1] for x in real_new_memory],
            "Size (GB)": [x[2] for x in real_new_memory],
            "First Python Frame": [x[3] for x in real_new_memory], # this is the first frame that shows a `.py` file
            "Full Stack Trace": [x[3] for x in real_new_memory], 
            "Pruned Stack Trace": [prune_frames(x[3]) for x in real_new_memory], # prune the stack trace
        }
        df_new = pd.DataFrame(data_new)
        csv_new_path = os.path.join(mem_snapshot_csv_dir, f"activation_new_memory_peak_{i}_{i+1}.csv")
        df_new.to_csv(csv_new_path, index=False)
        logging.info(f"3.{i}-{i+1}.new: Exported to {csv_new_path}")

        data_gone = {
            "Addr (Decimal)": [x[0] for x in real_gone_memory],
            "Time of Allocation (us)": [x[1] for x in real_gone_memory],
            "Size (GB)": [x[2] for x in real_gone_memory],
            "First Python Frame": [x[3] for x in real_gone_memory], # this is the first frame that shows a `.py` file
            "Full Stack Trace": [x[3] for x in real_gone_memory], 
            "Pruned Stack Trace": [prune_frames(x[3]) for x in real_gone_memory], # prune the stack trace
        }
        df_gone = pd.DataFrame(data_gone)
        csv_gone_path = os.path.join(mem_snapshot_csv_dir, f"activation_gone_memory_peak_{i}_{i+1}.csv")
        df_gone.to_csv(csv_gone_path, index=False)
        logging.info(f"3.{i}-{i+1}.gone: Exported to {csv_gone_path}")

    
    # Compare the last micro-batches of each global batch
    logging.info(f"===== Compare Alive Memory between the last micro-batches of each global batch: peak[{NUM_MB-1}], peak[{2*NUM_MB-1}] =====")
    time_peak_lmb_1 = time_us_history[peaks[NUM_MB-1]] # last micro-batch of the first global batch
    time_peak_lmb_2 = time_us_history[peaks[2*NUM_MB-1]] # last micro-batch of the second global batch
    (new_memory, gone_memory), (real_new_memory, real_gone_memory), unchanged_memory = compare_alive_memory(tracker, time_peak_lmb_1, time_peak_lmb_2)
    logging.info(f"Memory at peak[{NUM_MB-1}]: {alloc_memory_history[peaks[NUM_MB-1]]:.5f}; Memory at peak[{2*NUM_MB-1}]: {alloc_memory_history[peaks[2*NUM_MB-1]]:.5f}; Delta: {alloc_memory_history[peaks[2*NUM_MB-1]] - alloc_memory_history[peaks[NUM_MB-1]]:.5f}")
    print_memory_list_summary(real_new_memory, "real_new_memory")
    print_memory_list_summary(real_gone_memory, "real_gone_memory")

    # export real_new_memory and real_gone_memory to csv
    logging.info(f"===== Export to CSV: new_memory and gone_memory between peak[{NUM_MB-1}] and peak[{2*NUM_MB-1}] =====")
    data_new = {
        "Addr (Decimal)": [x[0] for x in real_new_memory],
        "Time of Allocation (us)": [x[1] for x in real_new_memory],
        "Size (GB)": [x[2] for x in real_new_memory],
        "First Python Frame": [x[3] for x in real_new_memory], # this is the first frame that shows a `.py` file
        "Full Stack Trace": [x[3] for x in real_new_memory], 
        "Pruned Stack Trace": [prune_frames(x[3]) for x in real_new_memory], # prune the stack trace
    }
    df_new = pd.DataFrame(data_new)
    csv_new_path = os.path.join(mem_snapshot_csv_dir, f"activation_new_memory_peak_{NUM_MB-1}_{2*NUM_MB-1}.csv")
    df_new.to_csv(csv_new_path, index=False)
    logging.info(f"3.{NUM_MB-1}-{2*NUM_MB-1}.new: Exported to {csv_new_path}")

    data_gone = {
        "Addr (Decimal)": [x[0] for x in real_gone_memory],
        "Time of Allocation (us)": [x[1] for x in real_gone_memory],
        "Size (GB)": [x[2] for x in real_gone_memory],
        "First Python Frame": [x[3] for x in real_gone_memory], # this is the first frame that shows a `.py` file
        "Full Stack Trace": [x[3] for x in real_gone_memory], 
        "Pruned Stack Trace": [prune_frames(x[3]) for x in real_gone_memory], # prune the stack trace
    }
    df_gone = pd.DataFrame(data_gone)
    csv_gone_path = os.path.join(mem_snapshot_csv_dir, f"activation_gone_memory_peak_{NUM_MB-1}_{2*NUM_MB-1}.csv")
    df_gone.to_csv(csv_gone_path, index=False)
    logging.info(f"3.{NUM_MB-1}-{2*NUM_MB-1}.gone: Exported to {csv_gone_path}")