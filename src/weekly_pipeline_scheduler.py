"""Weekly Pipeline Scheduler - Automates weekly pipeline re-runs.

This module schedules and executes the full pipeline weekly,
comparing new patterns with old and generating reports.
"""
import json
import logging
import subprocess
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class WeeklyPipelineScheduler:
    """Schedules and manages weekly pipeline executions."""
    
    def __init__(self, data_dir: str = 'data', backup_dir: str = 'backups'):
        self.data_dir = Path(data_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        self.schedule_file = self.data_dir / 'pipeline_schedule.json'
        self.schedule = self._load_schedule()
        
        logger.info(f"PipelineScheduler initialized. Last run: {self.schedule.get('last_run', 'Never')}")
    
    def _load_schedule(self) -> Dict:
        """Load schedule configuration."""
        if self.schedule_file.exists():
            try:
                with open(self.schedule_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading schedule: {e}")
        
        return {
            'created_at': datetime.now().isoformat(),
            'last_run': None,
            'run_history': [],
            'schedule': 'weekly',  # weekly, daily, monthly
            'day_of_week': 6,  # Sunday (0=Monday, 6=Sunday)
            'hour': 2,  # 2 AM
            'enabled': True
        }
    
    def _save_schedule(self):
        """Save schedule configuration."""
        try:
            with open(self.schedule_file, 'w') as f:
                json.dump(self.schedule, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving schedule: {e}")
    
    def should_run(self) -> bool:
        """Check if pipeline should run now."""
        if not self.schedule.get('enabled', True):
            return False
        
        last_run = self.schedule.get('last_run')
        if not last_run:
            return True
        
        last_run_dt = datetime.fromisoformat(last_run)
        now = datetime.now()
        
        # Check if it's been at least 7 days since last run
        days_since_last_run = (now - last_run_dt).days
        
        # Also check if it's the scheduled day and time
        is_scheduled_day = now.weekday() == self.schedule.get('day_of_week', 6)
        is_after_scheduled_hour = now.hour >= self.schedule.get('hour', 2)
        
        return days_since_last_run >= 7 or (is_scheduled_day and is_after_scheduled_hour and days_since_last_run >= 6)
    
    def run_pipeline(self) -> Dict:
        """Execute the full pipeline.
        
        Returns:
            Run result dict
        """
        result = {
            'started_at': datetime.now().isoformat(),
            'completed_at': None,
            'success': False,
            'error': None,
            'backup_path': None,
            'comparison': None
        }
        
        try:
            # 1. Backup current data
            backup_path = self._backup_current_data()
            result['backup_path'] = str(backup_path)
            logger.info(f"Backed up data to {backup_path}")
            
            # 2. Run pipeline
            logger.info("Starting pipeline execution...")
            process = subprocess.run(
                ['python', 'main.py'],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if process.returncode != 0:
                result['error'] = f"Pipeline failed: {process.stderr}"
                logger.error(result['error'])
                return result
            
            result['success'] = True
            result['completed_at'] = datetime.now().isoformat()
            
            # 3. Sync pattern IDs to preserve stable IDs
            logger.info("Syncing pattern IDs...")
            try:
                from pattern_id_manager import PatternIDManager
                manager = PatternIDManager()
                
                with open(self.data_dir / 'patterns.json', 'r') as f:
                    new_patterns = json.load(f)
                
                synced_patterns, sync_report = manager.sync_patterns(new_patterns)
                
                # Save synced patterns with stable IDs
                with open(self.data_dir / 'patterns.json', 'w') as f:
                    json.dump(synced_patterns, f, indent=2)
                
                result['pattern_sync'] = sync_report
                logger.info(f"Pattern sync: {sync_report['matched_existing']} preserved, "
                           f"{sync_report['new_discoveries']} new, {sync_report['archived']} archived")
            except Exception as e:
                logger.error(f"Pattern ID sync failed: {e}")
            
            # 4. Compare with previous patterns
            if backup_path:
                comparison = self._compare_patterns(backup_path)
                result['comparison'] = comparison
            
            # 5. Update schedule
            self.schedule['last_run'] = datetime.now().isoformat()
            self.schedule['run_history'].append({
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'backup_path': str(backup_path)
            })
            # Keep last 10 runs
            self.schedule['run_history'] = self.schedule['run_history'][-10:]
            self._save_schedule()
            
            logger.info("Pipeline completed successfully")
            
        except subprocess.TimeoutExpired:
            result['error'] = "Pipeline timed out after 1 hour"
            logger.error(result['error'])
        except Exception as e:
            result['error'] = f"Pipeline error: {str(e)}"
            logger.error(result['error'])
        
        return result
    
    def _backup_current_data(self) -> Path:
        """Backup current data directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f'data_backup_{timestamp}'
        backup_path.mkdir(exist_ok=True)
        
        # Copy key files
        files_to_backup = [
            'patterns.json',
            'features_matrix.csv',
            'trade_outcomes.json',
            'open_trades.json'
        ]
        
        for filename in files_to_backup:
            src = self.data_dir / filename
            if src.exists():
                shutil.copy2(src, backup_path / filename)
        
        # Copy occurrence files
        for occ_file in self.data_dir.glob('pattern_*_occurrences.json'):
            shutil.copy2(occ_file, backup_path / occ_file.name)
        
        return backup_path
    
    def _compare_patterns(self, backup_path: Path) -> Dict:
        """Compare new patterns with backed up patterns.
        
        Returns:
            Comparison dict
        """
        comparison = {
            'old_count': 0,
            'new_count': 0,
            'added': [],
            'removed': [],
            'performance_changes': []
        }
        
        try:
            # Load old patterns
            old_patterns_file = backup_path / 'patterns.json'
            if old_patterns_file.exists():
                with open(old_patterns_file, 'r') as f:
                    old_patterns = json.load(f)
                comparison['old_count'] = len(old_patterns)
            
            # Load new patterns
            new_patterns_file = self.data_dir / 'patterns.json'
            if new_patterns_file.exists():
                with open(new_patterns_file, 'r') as f:
                    new_patterns = json.load(f)
                comparison['new_count'] = len(new_patterns)
            
            # Compare (simplified - by pattern characteristics)
            old_keys = set(self._pattern_key(p) for p in old_patterns)
            new_keys = set(self._pattern_key(p) for p in new_patterns)
            
            comparison['added'] = list(new_keys - old_keys)[:10]  # Limit to 10
            comparison['removed'] = list(old_keys - new_keys)[:10]
            
            logger.info(f"Pattern comparison: {comparison['old_count']} old, {comparison['new_count']} new")
            
        except Exception as e:
            logger.error(f"Error comparing patterns: {e}")
        
        return comparison
    
    def _pattern_key(self, pattern: Dict) -> str:
        """Generate a key for pattern comparison."""
        # Use conditions as unique identifier
        conditions = pattern.get('conditions', {})
        if not conditions:
            return str(pattern.get('id', 'unknown'))
        
        # Sort conditions for consistent key
        cond_str = ','.join(f"{k}={v.get('operator','')}{v.get('value','')[:10]}" 
                          for k, v in sorted(conditions.items()))
        return cond_str[:100]  # Limit length
    
    def get_next_run(self) -> Optional[datetime]:
        """Get datetime of next scheduled run."""
        now = datetime.now()
        
        # Find next occurrence of scheduled day
        days_ahead = self.schedule.get('day_of_week', 6) - now.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        
        next_run = now + timedelta(days=days_ahead)
        next_run = next_run.replace(hour=self.schedule.get('hour', 2), minute=0, second=0)
        
        return next_run
    
    def get_run_history(self) -> List[Dict]:
        """Get history of pipeline runs."""
        return self.schedule.get('run_history', [])


def main():
    """Run scheduler check and execute if needed."""
    logging.basicConfig(level=logging.INFO)
    
    scheduler = WeeklyPipelineScheduler()
    
    print("\n" + "="*60)
    print("WEEKLY PIPELINE SCHEDULER")
    print("="*60)
    print(f"Last run: {scheduler.schedule.get('last_run', 'Never')}")
    print(f"Next scheduled: {scheduler.get_next_run()}")
    print(f"Should run now: {scheduler.should_run()}")
    
    if scheduler.should_run():
        print("\nStarting pipeline execution...")
        result = scheduler.run_pipeline()
        
        if result['success']:
            print("✓ Pipeline completed successfully")
            if result['comparison']:
                comp = result['comparison']
                print(f"\nPattern changes:")
                print(f"  Old patterns: {comp['old_count']}")
                print(f"  New patterns: {comp['new_count']}")
                print(f"  Added: {len(comp['added'])}")
                print(f"  Removed: {len(comp['removed'])}")
        else:
            print(f"✗ Pipeline failed: {result['error']}")
    else:
        print("\nPipeline not due for execution yet.")


if __name__ == '__main__':
    main()
