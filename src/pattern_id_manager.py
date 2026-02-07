"""Pattern ID Manager - Ensures stable pattern IDs across pipeline runs.

This module manages pattern identity persistence so that:
1. Patterns keep stable IDs across pipeline re-runs
2. Open trades remain linked to correct patterns
3. Trade history stays connected to pattern definitions
4. Removed patterns are archived, not lost
"""
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PatternIDManager:
    """Manages stable pattern IDs across pipeline runs."""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.pattern_registry_file = self.data_dir / 'pattern_registry.json'
        self.archive_file = self.data_dir / 'pattern_archive.json'
        
        # Load or create registry
        self.registry = self._load_registry()
        self.archive = self._load_archive()
        
        logger.info(f"PatternIDManager: {len(self.registry)} active, {len(self.archive)} archived")
    
    def _load_registry(self) -> Dict:
        """Load pattern registry."""
        if self.pattern_registry_file.exists():
            try:
                with open(self.pattern_registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
        return {'patterns': {}, 'next_id': 0}
    
    def _load_archive(self) -> List[Dict]:
        """Load archived patterns."""
        if self.archive_file.exists():
            try:
                with open(self.archive_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading archive: {e}")
        return []
    
    def _save_registry(self):
        """Save pattern registry."""
        try:
            with open(self.pattern_registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def _save_archive(self):
        """Save archived patterns."""
        try:
            with open(self.archive_file, 'w') as f:
                json.dump(self.archive, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving archive: {e}")
    
    def _generate_fingerprint(self, pattern: Dict) -> str:
        """Generate unique fingerprint for a pattern based on its conditions.
        
        This allows us to identify the same pattern even after re-discovery.
        """
        conditions = pattern.get('conditions', {})
        
        # Create canonical representation of conditions
        cond_parts = []
        for feature in sorted(conditions.keys()):
            cond = conditions[feature]
            part = f"{feature}:{cond.get('operator','')}:{cond.get('value',''):.6f}"
            cond_parts.append(part)
        
        # Add direction and label
        direction = pattern.get('direction', 'unknown')
        label = pattern.get('label_col', 'unknown')
        
        fingerprint_str = f"{direction}|{label}|{','.join(cond_parts)}"
        
        # Hash for compactness
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:16]
    
    def sync_patterns(self, new_patterns: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Sync new patterns with registry, preserving stable IDs.
        
        Args:
            new_patterns: Patterns from new pipeline run
        
        Returns:
            (patterns_with_stable_ids, sync_report)
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'new_patterns': len(new_patterns),
            'matched_existing': 0,
            'new_discoveries': 0,
            'archived': 0,
            'id_mapping': {}
        }
        
        # Track which existing patterns are still active
        active_fingerprints = set()
        synced_patterns = []
        
        for pattern in new_patterns:
            fingerprint = self._generate_fingerprint(pattern)
            active_fingerprints.add(fingerprint)
            
            # Check if this pattern already exists
            existing_id = None
            for pid, pdata in self.registry['patterns'].items():
                if pdata.get('fingerprint') == fingerprint:
                    existing_id = int(pid)
                    break
            
            if existing_id is not None:
                # Pattern exists - preserve ID and update stats
                pattern['id'] = existing_id
                
                # Preserve live stats from previous runs
                existing_data = self.registry['patterns'][str(existing_id)]
                if 'live_stats' in existing_data:
                    pattern['live_stats'] = existing_data['live_stats']
                if 'status' in existing_data:
                    pattern['status'] = existing_data['status']
                if 'disabled_reason' in existing_data:
                    pattern['disabled_reason'] = existing_data['disabled_reason']
                
                # Update registry with new discovery info
                self.registry['patterns'][str(existing_id)]['last_seen'] = datetime.now().isoformat()
                self.registry['patterns'][str(existing_id)]['discovery_count'] = \
                    existing_data.get('discovery_count', 0) + 1
                
                report['matched_existing'] += 1
                report['id_mapping'][existing_id] = existing_id
                
            else:
                # New pattern - assign new ID
                new_id = self.registry['next_id']
                self.registry['next_id'] += 1
                
                pattern['id'] = new_id
                
                # Register new pattern
                self.registry['patterns'][str(new_id)] = {
                    'fingerprint': fingerprint,
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat(),
                    'discovery_count': 1,
                    'direction': pattern.get('direction'),
                    'label_col': pattern.get('label_col')
                }
                
                report['new_discoveries'] += 1
                report['id_mapping'][new_id] = new_id
            
            synced_patterns.append(pattern)
        
        # Archive patterns that no longer exist
        current_ids = {str(p['id']) for p in synced_patterns}
        archived_count = 0
        
        for pid, pdata in list(self.registry['patterns'].items()):
            if pid not in current_ids:
                # Pattern no longer active - archive it
                archived_pattern = {
                    'id': int(pid),
                    'fingerprint': pdata.get('fingerprint'),
                    'first_seen': pdata.get('first_seen'),
                    'last_seen': pdata.get('last_seen'),
                    'discovery_count': pdata.get('discovery_count', 0),
                    'archived_at': datetime.now().isoformat(),
                    'live_stats': pdata.get('live_stats', {}),
                    'final_status': pdata.get('status', 'ACTIVE')
                }
                self.archive.append(archived_pattern)
                archived_count += 1
                
                # Remove from active registry
                del self.registry['patterns'][pid]
        
        report['archived'] = archived_count
        
        # Save updates
        self._save_registry()
        self._save_archive()
        
        logger.info(f"Pattern sync complete: {report['matched_existing']} matched, "
                   f"{report['new_discoveries']} new, {report['archived']} archived")
        
        return synced_patterns, report
    
    def get_pattern_history(self, pattern_id: int) -> Optional[Dict]:
        """Get full history of a pattern (active or archived)."""
        # Check active patterns
        if str(pattern_id) in self.registry['patterns']:
            return {
                'status': 'active',
                'data': self.registry['patterns'][str(pattern_id)]
            }
        
        # Check archived patterns
        for archived in self.archive:
            if archived.get('id') == pattern_id:
                return {
                    'status': 'archived',
                    'data': archived
                }
        
        return None
    
    def get_all_active_ids(self) -> List[int]:
        """Get list of all active pattern IDs."""
        return [int(pid) for pid in self.registry['patterns'].keys()]
    
    def is_active(self, pattern_id: int) -> bool:
        """Check if a pattern ID is currently active."""
        return str(pattern_id) in self.registry['patterns']


def integrate_with_pipeline():
    """Integration function to be called at end of pipeline."""
    import sys
    sys.path.insert(0, 'src')
    
    # Load newly discovered patterns
    with open('data/patterns.json', 'r') as f:
        new_patterns = json.load(f)
    
    # Sync with registry
    manager = PatternIDManager()
    synced_patterns, report = manager.sync_patterns(new_patterns)
    
    # Save synced patterns back
    with open('data/patterns.json', 'w') as f:
        json.dump(synced_patterns, f, indent=2)
    
    # Save sync report
    with open('data/pattern_sync_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nPattern ID Sync Complete:")
    print(f"  Active patterns: {len(synced_patterns)}")
    print(f"  Preserved IDs: {report['matched_existing']}")
    print(f"  New patterns: {report['new_discoveries']}")
    print(f"  Archived: {report['archived']}")
    
    return synced_patterns


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    integrate_with_pipeline()
