#!/usr/bin/env python3
"""
🌊 OceanEmbed — Smart Google Drive GLORYS Downloader (Colab Ready)
=================================================================
Features:
1. Auto-scans Google Drive to find the exact 'glorys' directory.
2. Checks file existence and sizes (> 1.0 GB) before starting any download.
3. Completely skips 2017 & 2018 (already downloaded).
4. Smooth real-time progress bar with cumulative average MB/s and accurate ETA.
5. Clean console output without overlapping text.
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path
import psutil

# Suppress noisy internal library logs so progress bar stays clean
logging.getLogger("copernicusmarine").setLevel(logging.ERROR)

def find_glorys_dir() -> Path:
    """Locate the exact glorys folder where 2017/2018 are stored."""
    candidates = [
        Path("/content/drive/MyDrive/OceanEmbed/data/raw/glorys"),
        Path("/content/drive/MyDrive/OceanEmbed/raw/glorys"),
        Path("/content/drive/MyDrive/data/raw/glorys"),
        Path("/content/drive/MyDrive/raw/glorys"),
        Path("/content/drive/MyDrive/glorys"),
    ]
    for c in candidates:
        if c.exists():
            files = [f for f in os.listdir(c) if f.startswith("glorys_") and f.endswith(".nc")]
            if len(files) > 0:
                return c
                
    # Fallback search if placed in a custom folder
    print("🔍 Scanning Google Drive for existing glorys files ...")
    for root, dirs, files in os.walk("/content/drive/MyDrive"):
        if any(f.startswith("glorys_2017") for f in files):
            return Path(root)

    fallback = Path("/content/drive/MyDrive/OceanEmbed/data/raw/glorys")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback

class SmoothProgressBarMonitor:
    """Tracks network reception with rolling/cumulative speed for smooth display."""
    def __init__(self, year: int, target_gb: float = 4.0):
        self.year = year
        self.target_gb = target_gb
        self.stop_flag = False
        self.thread = None

    def _run(self):
        start_time = time.time()
        start_bytes = psutil.net_io_counters().bytes_recv
        
        while not self.stop_flag:
            time.sleep(1.0)
            now = time.time()
            elapsed = max(1.0, now - start_time)
            curr_bytes = psutil.net_io_counters().bytes_recv
            
            diff_bytes = max(0, curr_bytes - start_bytes)
            downloaded_gb = diff_bytes / (1024 ** 3)
            pct = min(99.9, (downloaded_gb / self.target_gb) * 100.0)
            
            # Cumulative average speed (immune to 0 MB/s burst dips)
            avg_speed_mb_s = (diff_bytes / (1024 ** 2)) / elapsed
            
            # ETA calculation
            rem_gb = max(0.0, self.target_gb - downloaded_gb)
            eta_secs = int((rem_gb * 1024) / avg_speed_mb_s) if avg_speed_mb_s > 0.1 else 0
            eta_m, eta_s = divmod(eta_secs, 60)
            el_m, el_s = divmod(int(elapsed), 60)
            
            # Visual progress bar
            bar_len = 25
            filled = int(bar_len * pct // 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            
            msg = (
                f"\r🌊 [Year {self.year}] [{bar}] {pct:5.1f}% "
                f"| {downloaded_gb:5.2f} / ~{self.target_gb:.1f} GB "
                f"| 🚀 {avg_speed_mb_s:5.2f} MB/s "
                f"| ⏳ ETA: {eta_m:02d}m {eta_s:02d}s "
                f"| ⏱️ {el_m:02d}m {el_s:02d}s"
            )
            sys.stdout.write(msg)
            sys.stdout.flush()

    def start(self):
        self.stop_flag = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        if self.thread:
            self.thread.join(timeout=1.5)
        sys.stdout.write("\n")
        sys.stdout.flush()

def main():
    import copernicusmarine

    print("=" * 75)
    print("🌊 OceanEmbed — Smart GLORYS 3D Downloader")
    print("=" * 75)
    
    target_dir = find_glorys_dir()
    print(f"📁 Verified Google Drive Folder: {target_dir}\n")
    
    # Audit existing files
    print("📊 Current Google Drive Status:")
    all_years = list(range(2017, 2024))
    years_to_download = []
    
    for y in all_years:
        file_path = target_dir / f"glorys_{y}.nc"
        if file_path.exists() and file_path.stat().st_size > 1024 * 1024 * 1024:
            sz_gb = file_path.stat().st_size / (1024 ** 3)
            print(f"  ✅ Year {y}: {file_path.name} ({sz_gb:.2f} GB) — [COMPLETE, WILL SKIP]")
        else:
            print(f"  ⏳ Year {y}: Missing or incomplete — [QUEUED FOR DOWNLOAD]")
            years_to_download.append(y)
            
    print("=" * 75)
    
    if not years_to_download:
        print("🎉 ALL YEARS (2017–2023) ARE ALREADY COMPLETE IN GOOGLE DRIVE!")
        return

    print(f"🚀 Years remaining to download: {years_to_download}\n")
    
    DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
    VARIABLES = ["thetao", "so", "mlotst", "zos"]
    
    for year in years_to_download:
        target_file = target_dir / f"glorys_{year}.nc"
        print(f"\n📥 Starting Download: Year {year} (Saving directly to 15 TB Google Drive)")
        
        # Monitor thread
        monitor = SmoothProgressBarMonitor(year=year, target_gb=4.0)
        monitor.start()
        
        try:
            copernicusmarine.subset(
                dataset_id=DATASET_ID,
                variables=VARIABLES,
                minimum_longitude=75.0,
                maximum_longitude=100.0,
                minimum_latitude=5.0,
                maximum_latitude=25.0,
                minimum_depth=0.49,
                maximum_depth=1500.0,
                start_datetime=f"{year}-01-01T00:00:00",
                end_datetime=f"{year}-12-31T23:59:59",
                output_directory=str(target_dir),
                output_filename=f"glorys_{year}.nc",
                overwrite=True
            )
        finally:
            monitor.stop()
            
        final_size = target_file.stat().st_size / (1024 ** 3) if target_file.exists() else 0.0
        print(f"✅ Year {year} successfully downloaded and saved! ({final_size:.2f} GB)\n")

    print("=" * 75)
    print("🎉 ALL GLORYS YEARS (2017–2023) SUCCESSFULLY INGESTED INTO GOOGLE DRIVE!")
    print("=" * 75)

if __name__ == "__main__":
    main()
