"""
Validation Utilities

Validation utilities and helper functions.
"""

from typing import Dict, Any, Optional, List
import re
import logging

logger = logging.getLogger(__name__)


class ValidationUtils:
    """Utility functions for validation operations"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format"""
        pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def validate_json(json_str: str) -> bool:
        """Validate JSON format"""
        try:
            import json
            json.loads(json_str)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage"""
        # Remove or replace unsafe characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        return sanitized.strip()
    
    @staticmethod
    def validate_config(config: Dict[str, Any], required_keys: List[str]) -> Dict[str, Any]:
        """Validate configuration dictionary"""
        issues = []
        
        for key in required_keys:
            if key not in config:
                issues.append(f"Missing required key: {key}")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues
        }
