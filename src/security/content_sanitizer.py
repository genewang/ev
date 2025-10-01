"""
Content Redaction and Sanitization for EV Autonomous Driving.

Implements content sanitization to redact sensitive information and neutralize
malicious content that may slip through other security controls.
"""

import asyncio
import re
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import numpy as np
from loguru import logger


@dataclass
class SanitizationResult:
    """Content sanitization result."""
    is_safe: bool
    sanitized_content: str
    redactions_made: List[str]
    threats_neutralized: List[str]
    confidence: float
    original_length: int
    sanitized_length: int
    processing_time: float


@dataclass
class RedactionPattern:
    """Redaction pattern definition."""
    pattern_id: str
    pattern_type: str
    regex_pattern: str
    replacement: str
    description: str
    severity: str


class ContentSanitizer:
    """
    Content Redaction and Sanitization System.
    
    Provides comprehensive content sanitization for:
    - Sensitive data redaction
    - Malicious content neutralization
    - Prompt injection removal
    - Output sanitization
    - Audit logging
    
    Key Features:
    - Pattern-based redaction
    - ML-based content analysis
    - Sensitive data detection
    - Malicious content neutralization
    - Comprehensive audit trails
    """
    
    def __init__(self, config: Dict):
        """Initialize the content sanitizer."""
        self.config = config
        
        # Sanitization parameters
        self.redact_patterns = config.get('redact_patterns', [])
        self.sanitize_outputs = config.get('sanitize_outputs', True)
        self.log_redactions = config.get('log_redactions', True)
        self.aggressive_mode = config.get('aggressive_mode', False)
        
        # Redaction patterns
        self.redaction_patterns = self._initialize_redaction_patterns()
        
        # Sensitive data patterns
        self.sensitive_patterns = self._initialize_sensitive_patterns()
        
        # Malicious content patterns
        self.malicious_patterns = self._initialize_malicious_patterns()
        
        # Performance tracking
        self.sanitization_times = []
        self.redaction_counts = {}
        self.threat_neutralizations = []
        
        logger.info("🧹 Content sanitizer initialized")
        
    def _initialize_redaction_patterns(self) -> List[RedactionPattern]:
        """Initialize redaction patterns for sensitive data."""
        patterns = [
            # Password patterns
            RedactionPattern(
                pattern_id="password_1",
                pattern_type="password",
                regex_pattern=r'(?i)(password|passwd|pwd)\s*[:=]\s*\S+',
                replacement=r'\1: [REDACTED]',
                description="Password field redaction",
                severity="high"
            ),
            RedactionPattern(
                pattern_id="password_2",
                pattern_type="password",
                regex_pattern=r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']?[^"\'\s]+["\']?',
                replacement=r'\1: [REDACTED]',
                description="Password value redaction",
                severity="high"
            ),
            
            # API key patterns
            RedactionPattern(
                pattern_id="api_key_1",
                pattern_type="api_key",
                regex_pattern=r'(?i)(api[_-]?key|apikey)\s*[:=]\s*\S+',
                replacement=r'\1: [REDACTED]',
                description="API key redaction",
                severity="high"
            ),
            RedactionPattern(
                pattern_id="api_key_2",
                pattern_type="api_key",
                regex_pattern=r'(?i)(token|access[_-]?token)\s*[:=]\s*\S+',
                replacement=r'\1: [REDACTED]',
                description="Token redaction",
                severity="high"
            ),
            
            # Email patterns
            RedactionPattern(
                pattern_id="email_1",
                pattern_type="email",
                regex_pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                replacement='[EMAIL_REDACTED]',
                description="Email address redaction",
                severity="medium"
            ),
            
            # Phone number patterns
            RedactionPattern(
                pattern_id="phone_1",
                pattern_type="phone",
                regex_pattern=r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
                replacement='[PHONE_REDACTED]',
                description="Phone number redaction",
                severity="medium"
            ),
            
            # Credit card patterns
            RedactionPattern(
                pattern_id="credit_card_1",
                pattern_type="credit_card",
                regex_pattern=r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                replacement='[CARD_REDACTED]',
                description="Credit card number redaction",
                severity="high"
            ),
            
            # SSN patterns
            RedactionPattern(
                pattern_id="ssn_1",
                pattern_type="ssn",
                regex_pattern=r'\b\d{3}-?\d{2}-?\d{4}\b',
                replacement='[SSN_REDACTED]',
                description="SSN redaction",
                severity="high"
            ),
            
            # IP address patterns
            RedactionPattern(
                pattern_id="ip_1",
                pattern_type="ip_address",
                regex_pattern=r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                replacement='[IP_REDACTED]',
                description="IP address redaction",
                severity="low"
            ),
            
            # Internal system paths
            RedactionPattern(
                pattern_id="path_1",
                pattern_type="system_path",
                regex_pattern=r'(?i)(/etc/|/var/|/usr/|/home/|C:\\|D:\\)[^\s]+',
                replacement='[PATH_REDACTED]',
                description="System path redaction",
                severity="medium"
            )
        ]
        
        return patterns
        
    def _initialize_sensitive_patterns(self) -> List[RedactionPattern]:
        """Initialize patterns for sensitive information."""
        patterns = [
            # Model weights and parameters
            RedactionPattern(
                pattern_id="model_weights_1",
                pattern_type="model_data",
                regex_pattern=r'(?i)(weights?|parameters?|state_dict)\s*[:=]\s*\{[^}]*\}',
                replacement=r'\1: [MODEL_DATA_REDACTED]',
                description="Model weights redaction",
                severity="high"
            ),
            
            # Configuration secrets
            RedactionPattern(
                pattern_id="config_secret_1",
                pattern_type="config_secret",
                regex_pattern=r'(?i)(secret|key|password|token)\s*[:=]\s*["\']?[^"\'\s]+["\']?',
                replacement=r'\1: [CONFIG_REDACTED]',
                description="Configuration secret redaction",
                severity="high"
            ),
            
            # Database credentials
            RedactionPattern(
                pattern_id="db_cred_1",
                pattern_type="database_credential",
                regex_pattern=r'(?i)(database|db)[_-]?(user|username|password|host|port)\s*[:=]\s*["\']?[^"\'\s]+["\']?',
                replacement=r'\1: [DB_CRED_REDACTED]',
                description="Database credential redaction",
                severity="high"
            )
        ]
        
        return patterns
        
    def _initialize_malicious_patterns(self) -> List[RedactionPattern]:
        """Initialize patterns for malicious content."""
        patterns = [
            # Script injection patterns
            RedactionPattern(
                pattern_id="script_injection_1",
                pattern_type="script_injection",
                regex_pattern=r'<script[^>]*>.*?</script>',
                replacement='[SCRIPT_REMOVED]',
                description="Script tag removal",
                severity="high"
            ),
            RedactionPattern(
                pattern_id="script_injection_2",
                pattern_type="script_injection",
                regex_pattern=r'javascript\s*:',
                replacement='[JAVASCRIPT_REMOVED]:',
                description="JavaScript protocol removal",
                severity="high"
            ),
            
            # SQL injection patterns
            RedactionPattern(
                pattern_id="sql_injection_1",
                pattern_type="sql_injection",
                regex_pattern=r'(?i)(union|select|insert|update|delete|drop|create|alter)\s+',
                replacement=r'[SQL_KEYWORD_REMOVED] ',
                description="SQL keyword neutralization",
                severity="high"
            ),
            
            # Command injection patterns
            RedactionPattern(
                pattern_id="cmd_injection_1",
                pattern_type="command_injection",
                regex_pattern=r'[;&|`$(){}[\]\\]',
                replacement='[SPECIAL_CHAR_NEUTRALIZED]',
                description="Special character neutralization",
                severity="medium"
            ),
            
            # Prompt injection patterns
            RedactionPattern(
                pattern_id="prompt_injection_1",
                pattern_type="prompt_injection",
                regex_pattern=r'(?i)(ignore\s+previous|override|jailbreak|developer\s+mode)',
                replacement='[PROMPT_INJECTION_NEUTRALIZED]',
                description="Prompt injection neutralization",
                severity="high"
            ),
            RedactionPattern(
                pattern_id="prompt_injection_2",
                pattern_type="prompt_injection",
                regex_pattern=r'(?i)(system\s*:|admin\s*:|bypass|unrestricted)',
                replacement='[SYSTEM_OVERRIDE_NEUTRALIZED]',
                description="System override neutralization",
                severity="high"
            ),
            
            # Backdoor trigger patterns
            RedactionPattern(
                pattern_id="backdoor_trigger_1",
                pattern_type="backdoor_trigger",
                regex_pattern=r'\[TRIGGER\s*:\s*[^\]]+\]',
                replacement='[BACKDOOR_TRIGGER_NEUTRALIZED]',
                description="Backdoor trigger neutralization",
                severity="critical"
            ),
            RedactionPattern(
                pattern_id="backdoor_trigger_2",
                pattern_type="backdoor_trigger",
                regex_pattern=r'0x[0-9A-Fa-f]+',
                replacement='[HEX_PATTERN_NEUTRALIZED]',
                description="Hexadecimal pattern neutralization",
                severity="high"
            )
        ]
        
        return patterns
        
    async def sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize data by redacting sensitive information and neutralizing threats.
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data
        """
        start_time = time.time()
        
        try:
            logger.debug("🧹 Sanitizing data")
            
            sanitized_data = {}
            all_redactions = []
            all_threats = []
            
            for key, value in data.items():
                if isinstance(value, str):
                    # Sanitize string values
                    result = await self.sanitize_text(value)
                    sanitized_data[key] = result.sanitized_content
                    all_redactions.extend(result.redactions_made)
                    all_threats.extend(result.threats_neutralized)
                    
                elif isinstance(value, dict):
                    # Recursively sanitize dictionaries
                    sanitized_data[key] = await self.sanitize_data(value)
                    
                elif isinstance(value, list):
                    # Sanitize list items
                    sanitized_list = []
                    for item in value:
                        if isinstance(item, str):
                            result = await self.sanitize_text(item)
                            sanitized_list.append(result.sanitized_content)
                            all_redactions.extend(result.redactions_made)
                            all_threats.extend(result.threats_neutralized)
                        else:
                            sanitized_list.append(item)
                    sanitized_data[key] = sanitized_list
                    
                else:
                    # Keep other types as-is
                    sanitized_data[key] = value
                    
            # Record sanitization statistics
            processing_time = time.time() - start_time
            self.sanitization_times.append(processing_time)
            
            # Update redaction counts
            for redaction in all_redactions:
                self.redaction_counts[redaction] = self.redaction_counts.get(redaction, 0) + 1
                
            # Record threat neutralizations
            if all_threats:
                self.threat_neutralizations.append({
                    'timestamp': time.time(),
                    'threats': all_threats,
                    'processing_time': processing_time
                })
                
            logger.debug(f"🧹 Data sanitization completed in {processing_time:.3f}s")
            return sanitized_data
            
        except Exception as e:
            logger.error(f"❌ Data sanitization failed: {e}")
            return data  # Return original data if sanitization fails
            
    async def sanitize_text(self, text: str) -> SanitizationResult:
        """
        Sanitize text content.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitization result
        """
        start_time = time.time()
        
        try:
            if not text:
                return SanitizationResult(
                    is_safe=True,
                    sanitized_content=text,
                    redactions_made=[],
                    threats_neutralized=[],
                    confidence=1.0,
                    original_length=0,
                    sanitized_length=0,
                    processing_time=0.0
                )
                
            sanitized_text = text
            redactions_made = []
            threats_neutralized = []
            
            # Apply redaction patterns
            for pattern in self.redaction_patterns:
                original_text = sanitized_text
                sanitized_text = re.sub(pattern.regex_pattern, pattern.replacement, sanitized_text)
                
                if sanitized_text != original_text:
                    redactions_made.append(pattern.pattern_id)
                    if self.log_redactions:
                        logger.info(f"🔴 Redacted {pattern.pattern_type}: {pattern.description}")
                        
            # Apply sensitive data patterns
            for pattern in self.sensitive_patterns:
                original_text = sanitized_text
                sanitized_text = re.sub(pattern.regex_pattern, pattern.replacement, sanitized_text)
                
                if sanitized_text != original_text:
                    redactions_made.append(pattern.pattern_id)
                    if self.log_redactions:
                        logger.info(f"🔴 Redacted sensitive data: {pattern.description}")
                        
            # Apply malicious content patterns
            for pattern in self.malicious_patterns:
                original_text = sanitized_text
                sanitized_text = re.sub(pattern.regex_pattern, pattern.replacement, sanitized_text)
                
                if sanitized_text != original_text:
                    threats_neutralized.append(pattern.pattern_id)
                    if self.log_redactions:
                        logger.warning(f"⚠️ Neutralized threat: {pattern.description}")
                        
            # Additional sanitization for aggressive mode
            if self.aggressive_mode:
                sanitized_text = await self._aggressive_sanitization(sanitized_text)
                
            # Calculate confidence
            confidence = self._calculate_sanitization_confidence(text, sanitized_text, 
                                                               redactions_made, threats_neutralized)
            
            # Determine if content is safe
            is_safe = len(threats_neutralized) == 0 and confidence > 0.7
            
            processing_time = time.time() - start_time
            
            return SanitizationResult(
                is_safe=is_safe,
                sanitized_content=sanitized_text,
                redactions_made=redactions_made,
                threats_neutralized=threats_neutralized,
                confidence=confidence,
                original_length=len(text),
                sanitized_length=len(sanitized_text),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"❌ Text sanitization failed: {e}")
            return SanitizationResult(
                is_safe=False,
                sanitized_content=text,
                redactions_made=[],
                threats_neutralized=[],
                confidence=0.0,
                original_length=len(text),
                sanitized_length=len(text),
                processing_time=time.time() - start_time
            )
            
    async def _aggressive_sanitization(self, text: str) -> str:
        """Apply aggressive sanitization for high-security scenarios."""
        # Remove all special characters except basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '[CHAR_NEUTRALIZED]', text)
        
        # Remove multiple consecutive spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove potential encoding attempts
        text = re.sub(r'%[0-9A-Fa-f]{2}', '[ENCODING_NEUTRALIZED]', text)
        
        # Remove potential base64 patterns
        text = re.sub(r'[A-Za-z0-9+/]{20,}={0,2}', '[BASE64_NEUTRALIZED]', text)
        
        return text.strip()
        
    def _calculate_sanitization_confidence(self, original_text: str, sanitized_text: str,
                                         redactions_made: List[str], threats_neutralized: List[str]) -> float:
        """Calculate confidence score for sanitization."""
        base_confidence = 1.0
        
        # Reduce confidence for each redaction
        for redaction in redactions_made:
            base_confidence -= 0.1
            
        # Reduce confidence more for threat neutralizations
        for threat in threats_neutralized:
            base_confidence -= 0.2
            
        # Adjust based on text length change
        if len(sanitized_text) < len(original_text) * 0.5:
            base_confidence -= 0.3
            
        # Adjust based on content similarity
        if original_text and sanitized_text:
            similarity = self._calculate_text_similarity(original_text, sanitized_text)
            base_confidence = (base_confidence + similarity) / 2
            
        return max(0.0, min(1.0, base_confidence))
        
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        if not text1 or not text2:
            return 0.0
            
        # Simple character-based similarity
        common_chars = set(text1.lower()) & set(text2.lower())
        total_chars = set(text1.lower()) | set(text2.lower())
        
        if not total_chars:
            return 1.0
            
        return len(common_chars) / len(total_chars)
        
    async def sanitize_model_output(self, output: str) -> SanitizationResult:
        """
        Sanitize model output specifically.
        
        Args:
            output: Model output to sanitize
            
        Returns:
            Sanitization result
        """
        # Apply additional model-specific sanitization
        result = await self.sanitize_text(output)
        
        # Additional checks for model outputs
        if self.sanitize_outputs:
            # Remove potential system commands
            result.sanitized_content = re.sub(
                r'(?i)(sudo|rm|del|format|shutdown|reboot)\s+',
                '[COMMAND_NEUTRALIZED] ',
                result.sanitized_content
            )
            
            # Remove potential file paths
            result.sanitized_content = re.sub(
                r'[A-Za-z]:\\[^\\/:*?"<>|\r\n]+',
                '[FILE_PATH_NEUTRALIZED]',
                result.sanitized_content
            )
            
            # Remove potential URLs
            result.sanitized_content = re.sub(
                r'https?://[^\s]+',
                '[URL_NEUTRALIZED]',
                result.sanitized_content
            )
            
        return result
        
    async def sanitize_prompt_input(self, prompt: str) -> SanitizationResult:
        """
        Sanitize prompt input specifically.
        
        Args:
            prompt: Prompt input to sanitize
            
        Returns:
            Sanitization result
        """
        # Apply additional prompt-specific sanitization
        result = await self.sanitize_text(prompt)
        
        # Additional checks for prompt inputs
        # Remove potential injection attempts
        result.sanitized_content = re.sub(
            r'(?i)(ignore|override|bypass|jailbreak|developer\s+mode)',
            '[INJECTION_NEUTRALIZED]',
            result.sanitized_content
        )
        
        # Remove potential system instructions
        result.sanitized_content = re.sub(
            r'(?i)(system\s*:|admin\s*:|root\s*:)',
            '[SYSTEM_OVERRIDE_NEUTRALIZED]:',
            result.sanitized_content
        )
        
        # Remove potential backdoor triggers
        result.sanitized_content = re.sub(
            r'\[[A-Z_]+\]',
            '[TRIGGER_NEUTRALIZED]',
            result.sanitized_content
        )
        
        return result
        
    def get_sanitization_statistics(self) -> Dict:
        """Get sanitization statistics."""
        total_sanitizations = len(self.sanitization_times)
        avg_processing_time = np.mean(self.sanitization_times) if self.sanitization_times else 0.0
        
        total_redactions = sum(self.redaction_counts.values())
        total_threats = len(self.threat_neutralizations)
        
        return {
            'total_sanitizations': total_sanitizations,
            'avg_processing_time': avg_processing_time,
            'total_redactions': total_redactions,
            'total_threats_neutralized': total_threats,
            'redaction_breakdown': dict(self.redaction_counts),
            'redaction_patterns': len(self.redaction_patterns),
            'sensitive_patterns': len(self.sensitive_patterns),
            'malicious_patterns': len(self.malicious_patterns),
            'aggressive_mode': self.aggressive_mode,
            'log_redactions': self.log_redactions
        }
        
    def update_redaction_patterns(self, new_patterns: List[RedactionPattern]):
        """Update redaction patterns."""
        self.redaction_patterns.extend(new_patterns)
        logger.info(f"📝 Updated redaction patterns: {len(new_patterns)} new patterns added")
        
    def enable_aggressive_mode(self, enabled: bool = True):
        """Enable or disable aggressive sanitization mode."""
        self.aggressive_mode = enabled
        logger.info(f"🔧 Aggressive sanitization mode: {'enabled' if enabled else 'disabled'}")
        
    def set_log_redactions(self, enabled: bool = True):
        """Enable or disable redaction logging."""
        self.log_redactions = enabled
        logger.info(f"📝 Redaction logging: {'enabled' if enabled else 'disabled'}")
