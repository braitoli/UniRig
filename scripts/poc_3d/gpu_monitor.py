#!/usr/bin/env python3
"""
GPU Resource Monitor: Ghi lại thông số VRAM, GPU Load, Nhiệt độ và Công suất vào file CSV
Được sử dụng trong các bài benchmark Trellis 2 và Pixal3D trên Vast.ai.
"""

import sys
import time
import subprocess
import csv
from datetime import datetime

def monitor(output_csv="gpu_metrics.csv", interval_sec=0.5, duration_sec=None):
    print(f"[*] Bắt đầu theo dõi GPU -> Lưu vào {output_csv} (interval: {interval_sec}s)")
    start_time = time.time()
    
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "elapsed_sec", "gpu_util_pct", 
            "mem_used_mb", "mem_total_mb", "mem_free_mb", 
            "temp_c", "power_draw_w"
        ])
        
        try:
            while True:
                now = time.time()
                elapsed = now - start_time
                if duration_sec and elapsed > duration_sec:
                    break
                
                cmd = [
                    "nvidia-smi", 
                    "--query-gpu=utilization.gpu,memory.used,memory.total,memory.free,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits"
                ]
                res = subprocess.check_output(cmd).decode("utf-8").strip().split(",")
                util, mem_used, mem_total, mem_free, temp, power = [x.strip() for x in res]
                
                writer.writerow([
                    datetime.now().isoformat(), round(elapsed, 2), util,
                    mem_used, mem_total, mem_free, temp, power
                ])
                f.flush()
                time.sleep(interval_sec)
        except KeyboardInterrupt:
            print("\n[*] Đã dừng theo dõi GPU.")

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "gpu_metrics.csv"
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else None
    monitor(output_csv=out_file, duration_sec=duration)
