# gpu-logger

## Install

```
conda env create --file environment.yml
```

## Usage

Save record with interval of 1 hour:

```
conda activate gpu-logger
python main.py main --path_out log.jsonl --interval 3600
```

Example record:

```
{
  "time": 1629639339.0095732,
  "devices": [
    {
      "id": 0,
      "name": "Quadro RTX 8000",
      "mem_used": 36.842,
      "mem_total": 48.601,
      "util": 0.99
    },
    {
      "id": 1,
      "name": "Quadro RTX 8000",
      "mem_used": 0.003,
      "mem_total": 48.598,
      "util": 0.0
    }
  ],
  "processes": [
    {
      "device_id": 0,
      "user": "student",
      "name": "/home/anaconda3/envs/bin/python train.py",
      "uptime": "1-06:05:35",
      "pid": 18751,
      "mem_used": 36.839
    }
  ]
}
```
