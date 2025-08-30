#!/usr/bin/env python3
"""
Automatic Data Updater for Gold Price Prediction System
This script runs in the background to automatically update data at specified intervals
"""

import time
import threading
import schedule
from datetime import datetime, timedelta
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_collector import GoldDataCollector

class AutoUpdater:
    def __init__(self, update_interval_minutes=30):
        """
        Initialize the auto updater
        
        Args:
            update_interval_minutes (int): How often to update data in minutes
        """
        self.update_interval = update_interval_minutes
        self.data_collector = GoldDataCollector()
        self.running = False
        self.last_update = None
        self.update_thread = None
        
    def should_update(self):
        """
        Check if data should be updated based on time interval
        """
        last_update_file = os.path.join('data', 'last_update.txt')
        
        if not os.path.exists(last_update_file):
            return True
            
        try:
            with open(last_update_file, 'r') as f:
                last_update_str = f.read().strip()
                last_update = datetime.fromisoformat(last_update_str)
                
            # Check if enough time has passed
            time_diff = datetime.now() - last_update
            return time_diff.total_seconds() >= (self.update_interval * 60)
            
        except Exception as e:
            print(f"Error checking last update: {e}")
            return True
    
    def update_data(self):
        """
        Perform data update
        """
        try:
            print(f"[{datetime.now()}] Starting automatic data update...")
            
            # Force refresh to get latest data
            result = self.data_collector.update_data(force_refresh=True)
            
            if result:
                print(f"[{datetime.now()}] Data update completed successfully")
                self.last_update = datetime.now()
                return True
            else:
                print(f"[{datetime.now()}] Data update returned no result")
                return False
                
        except Exception as e:
            print(f"[{datetime.now()}] Error during data update: {e}")
            return False
    
    def update_job(self):
        """
        Job function that checks if update is needed and performs it
        """
        if self.should_update():
            self.update_data()
        else:
            print(f"[{datetime.now()}] Data is up to date, skipping update")
    
    def start_scheduler(self):
        """
        Start the background scheduler
        """
        print(f"Starting auto updater with {self.update_interval} minute intervals")
        
        # Schedule the job
        schedule.every(self.update_interval).minutes.do(self.update_job)
        
        # Run initial update
        self.update_job()
        
        self.running = True
        
        # Run scheduler in background
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def start(self):
        """
        Start the auto updater in a background thread
        """
        if self.update_thread and self.update_thread.is_alive():
            print("Auto updater is already running")
            return
            
        self.update_thread = threading.Thread(target=self.start_scheduler, daemon=True)
        self.update_thread.start()
        print("Auto updater started in background thread")
    
    def stop(self):
        """
        Stop the auto updater
        """
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        print("Auto updater stopped")
    
    def get_status(self):
        """
        Get current status of the auto updater
        """
        return {
            'running': self.running,
            'update_interval_minutes': self.update_interval,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'next_update_in_minutes': self.get_next_update_time()
        }
    
    def get_next_update_time(self):
        """
        Calculate minutes until next update
        """
        if not self.last_update:
            return 0
            
        next_update = self.last_update + timedelta(minutes=self.update_interval)
        time_diff = next_update - datetime.now()
        
        if time_diff.total_seconds() <= 0:
            return 0
        
        return int(time_diff.total_seconds() / 60)

def main():
    """
    Main function to run the auto updater as a standalone script
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Gold Price Data Auto Updater')
    parser.add_argument('--interval', type=int, default=30, 
                       help='Update interval in minutes (default: 30)')
    parser.add_argument('--once', action='store_true',
                       help='Run update once and exit')
    
    args = parser.parse_args()
    
    updater = AutoUpdater(update_interval_minutes=args.interval)
    
    if args.once:
        # Run once and exit
        updater.update_data()
    else:
        # Run continuously
        try:
            updater.start_scheduler()
        except KeyboardInterrupt:
            print("\nStopping auto updater...")
            updater.stop()

if __name__ == '__main__':
    main()