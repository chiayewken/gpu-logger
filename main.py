"""
Log gpu usage statistics at regular intervals
Adapted from: https://github.com/fbcotter/py3nvml/blob/0.2.5/scripts/py3smi
"""

import json
import os
import pwd
import time
from pathlib import Path
from subprocess import Popen, PIPE
from typing import List

import matplotlib.pyplot as plt
from fire import Fire
from py3nvml.py3nvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetCount,
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetMinorNumber,
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetName,
)
from pydantic import BaseModel
from tqdm import tqdm


class Device(BaseModel):
    id: int
    name: str
    mem_used: float
    mem_total: float
    util: float


class Process(BaseModel):
    device_id: int
    user: str
    name: str
    uptime: str
    pid: int
    mem_used: float


def get_device(index: int) -> Device:
    h = nvmlDeviceGetHandleByIndex(index)
    mem_info = nvmlDeviceGetMemoryInfo(h)

    return Device(
        id=index,
        name=nvmlDeviceGetName(h),
        mem_used=(mem_info.used >> 20) / 1000,
        mem_total=(mem_info.total >> 20) / 1000,
        util=nvmlDeviceGetUtilizationRates(h).gpu / 100,
    )


def get_uname_pid(pid):
    try:
        # the /proc/PID is owned by process creator
        proc_stat_file = os.stat("/proc/%d" % pid)
        # # get UID via stat call
        uid = proc_stat_file.st_uid
        # # look up the username from uid
        username = pwd.getpwuid(uid)[0]
    except Exception as e:
        print(e)
        username = "???"
    return username


def get_pname(i):
    try:
        sess = Popen(["ps", "-o", "cmd=", "{}".format(i)], stdout=PIPE, stderr=PIPE)
        stdout, stderr = sess.communicate()
        name = stdout.decode("utf-8").strip()
    except Exception as e:
        print(e)
        name = ""
    return name


def get_uptime(pid):
    try:
        sess = Popen(["ps", "-q", str(pid), "-o", "etime="], stdout=PIPE, stderr=PIPE)
        stdout, stderr = sess.communicate()
        uptime = stdout.decode("utf-8").strip()
    except Exception as e:
        print(e)
        uptime = "?"
    return uptime


def get_processes(device: Device) -> List[Process]:
    h = nvmlDeviceGetHandleByIndex(device.id)
    procs = []

    min_number = nvmlDeviceGetMinorNumber(h)
    for raw in nvmlDeviceGetComputeRunningProcesses(h):
        p = Process(
            device_id=min_number,
            user=get_uname_pid(raw.pid),
            name=get_pname(raw.pid),
            uptime=get_uptime(raw.pid),
            pid=raw.pid,
            mem_used=(raw.usedGpuMemory >> 20) / 1000,
        )
        procs.append(p)

    return procs


class NvmlContext:
    def __init__(self):
        self.num_gpus: int = 0

    def __enter__(self):
        nvmlInit()
        self.num_gpus = nvmlDeviceGetCount()
        return self

    def __exit__(self, exit_type, value, traceback):
        nvmlShutdown()


def test_context():
    with NvmlContext() as c:
        print(dict(num_gpus=c.num_gpus))


def test_get_device():
    with NvmlContext() as c:
        for i in range(c.num_gpus):
            device = get_device(i)
            print(device.model_dump_json(indent=2))


def test_get_process():
    with NvmlContext() as c:
        for i in range(c.num_gpus):
            device = get_device(i)
            for process in get_processes(device):
                print(process.model_dump_json(indent=2))


class Record(BaseModel):
    time: float
    devices: List[Device]
    processes: List[Process]

    def get_device(self, i: int) -> Device:
        mapping = {device.id: device for device in self.devices}
        return mapping.get(i)


class LogData(BaseModel):
    records: List[Record]

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for record in self.records:
                f.write(record.model_dump_json() + "\n")

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            records = [Record(**json.loads(line)) for line in f.readlines()]
        return cls(records=records)


def save_logs(path_out: str, interval: int):
    data = LogData(records=[])
    for _ in tqdm(range(int(1e9))):
        with NvmlContext() as c:
            devices = [get_device(i) for i in range(c.num_gpus)]
            data.records.append(
                Record(
                    time=time.time(),
                    devices=devices,
                    processes=[p for d in devices for p in get_processes(d)],
                )
            )

        data.save(path_out)
        time.sleep(interval)


def visualize_logs(path: str, device: int, path_out: str):
    data = LogData.load(path)
    values = [record.get_device(device).mem_used for record in data.records]
    plt.plot(values)
    plt.savefig(path_out)


"""
python main.py save_logs logs.jsonl --interval 1
python main.py visualize_logs logs.jsonl --device 1 --path_out gpu_memory.png
"""

if __name__ == "__main__":
    Fire()
