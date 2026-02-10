"""
ICT Logging, Monitoring & Deployment System
============================================

Production-ready infrastructure for ICT algorithmic trading including:
- Structured logging for trades, errors, and system events
- Automatic trade journaling with analysis
- Health monitoring for connections, API limits, and resources
- Docker containerization
- Cloud deployment configurations (AWS/GCP)

LOGGING & MONITORING ARCHITECTURE:
=================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                   LOGGING, MONITORING & DEPLOYMENT                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      STRUCTURED LOGGING                               │    │
│  │                                                                       │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│  │  │  Trade Log   │  │  Error Log   │  │  System Log  │              │    │
│  │  │              │  │              │  │              │              │    │
│  │  │ • Entry/Exit │  │ • Exceptions │  │ • Startup    │              │    │
│  │  │ • P&L        │  │ • API Errors │  │ • Heartbeat  │              │    │
│  │  │ • Signals    │  │ • Timeouts   │  │ • Config     │              │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │    │
│  │                                                                       │    │
│  │  Outputs: Console │ File │ JSON │ Database │ Cloud (CloudWatch/GCP) │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      TRADE JOURNAL                                    │    │
│  │                                                                       │    │
│  │  For Each Trade:                                                      │    │
│  │  • Entry/Exit details          • Market context                      │    │
│  │  • ICT model used              • Confluence factors                  │    │
│  │  • Risk/Reward analysis        • Session/Kill zone                   │    │
│  │  • Performance metrics         • Lessons learned (AI)                │    │
│  │                                                                       │    │
│  │  Exports: Markdown │ HTML │ PDF │ CSV │ JSON                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      HEALTH MONITORING                                │    │
│  │                                                                       │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │    │
│  │  │  Broker     │  │  API Rate   │  │  System     │  │  Network  │  │    │
│  │  │  Connection │  │  Limits     │  │  Resources  │  │  Latency  │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │    │
│  │                                                                       │    │
│  │  Alerts: Email │ Telegram │ PagerDuty │ Slack                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      DEPLOYMENT                                       │    │
│  │                                                                       │    │
│  │  Docker:                        Cloud:                                │    │
│  │  • Multi-stage build            • AWS EC2/ECS/Lambda                 │    │
│  │  • Health checks                • GCP Compute/Cloud Run              │    │
│  │  • Volume mounts                • Auto-scaling                       │    │
│  │  • Environment config           • Load balancing                     │    │
│  │                                                                       │    │
│  │  CI/CD: GitHub Actions │ AWS CodePipeline │ GCP Cloud Build          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

Author: Claude (Anthropic)
Version: 1.0.0
"""

import logging
import logging.handlers
import json
import os
import sys
import time
import threading
import socket
import platform
import traceback
import gzip
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from collections import defaultdict, deque
from pathlib import Path
import hashlib
import csv
import io

# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class LogLevel(Enum):
    """Log levels"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogCategory(Enum):
    """Log categories for filtering"""
    TRADE = "trade"
    SIGNAL = "signal"
    ORDER = "order"
    POSITION = "position"
    RISK = "risk"
    SYSTEM = "system"
    ERROR = "error"
    PERFORMANCE = "performance"
    HEALTH = "health"
    AUDIT = "audit"


class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class DeploymentEnvironment(Enum):
    """Deployment environment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    correlation_id: str = ""
    trade_id: Optional[str] = None
    symbol: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.name,
            'category': self.category.value,
            'message': self.message,
            'data': self.data,
            'source': self.source,
            'correlation_id': self.correlation_id,
            'trade_id': self.trade_id,
            'symbol': self.symbol
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())


@dataclass
class TradeJournalEntry:
    """Complete trade journal entry"""
    # Trade identification
    trade_id: str
    symbol: str
    direction: str
    
    # Timing
    entry_time: datetime
    exit_time: Optional[datetime] = None
    duration_minutes: int = 0
    
    # Prices
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Position
    quantity: float = 0.0
    risk_amount: float = 0.0
    
    # Results
    pnl: float = 0.0
    pnl_pips: float = 0.0
    r_multiple: float = 0.0
    is_winner: bool = False
    exit_reason: str = ""
    
    # ICT Analysis
    model_name: str = ""
    confluence_factors: List[str] = field(default_factory=list)
    session: str = ""
    kill_zone: str = ""
    market_structure: str = ""
    
    # Context
    htf_bias: str = ""
    key_levels: List[float] = field(default_factory=list)
    news_events: List[str] = field(default_factory=list)
    
    # Performance metrics
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    efficiency_ratio: float = 0.0  # Actual profit / MFE
    
    # Notes
    pre_trade_notes: str = ""
    post_trade_notes: str = ""
    lessons_learned: str = ""
    rating: int = 0  # 1-5 self-rating
    
    # Metadata
    screenshots: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class HealthCheckResult:
    """Health check result"""
    name: str
    status: HealthStatus
    message: str
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'latency_ms': self.latency_ms,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_percent: float = 0.0
    disk_used_gb: float = 0.0
    disk_free_gb: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    open_files: int = 0
    threads: int = 0
    uptime_seconds: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class APIRateLimitStatus:
    """API rate limit status"""
    api_name: str
    requests_made: int
    requests_limit: int
    requests_remaining: int
    reset_time: datetime
    is_limited: bool = False
    
    @property
    def usage_percent(self) -> float:
        return (self.requests_made / self.requests_limit * 100) if self.requests_limit > 0 else 0


# =============================================================================
# STRUCTURED LOGGER
# =============================================================================

class StructuredLogger:
    """
    Production-ready structured logging system.
    Supports multiple outputs and formats.
    """
    
    def __init__(
        self,
        name: str = "ict_trading",
        log_dir: str = "logs",
        level: LogLevel = LogLevel.INFO,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = True,
        max_file_size_mb: int = 10,
        backup_count: int = 30,
        compress_backups: bool = True
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.level = level
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_json = enable_json
        self.max_file_size_mb = max_file_size_mb
        self.backup_count = backup_count
        self.compress_backups = compress_backups
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self._loggers: Dict[LogCategory, logging.Logger] = {}
        self._setup_loggers()
        
        # Correlation ID for request tracing
        self._correlation_id = threading.local()
        
        # Log buffer for batch operations
        self._buffer: deque = deque(maxlen=10000)
        self._buffer_lock = threading.Lock()
        
        # Metrics
        self._log_counts: Dict[str, int] = defaultdict(int)
    
    def _setup_loggers(self):
        """Set up loggers for each category"""
        # Common formatter
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        json_formatter = JsonFormatter()
        
        for category in LogCategory:
            logger = logging.getLogger(f"{self.name}.{category.value}")
            logger.setLevel(self.level.value)
            logger.handlers = []  # Clear existing handlers
            
            # Console handler
            if self.enable_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(detailed_formatter)
                console_handler.setLevel(self.level.value)
                logger.addHandler(console_handler)
            
            # File handler (rotating)
            if self.enable_file:
                file_path = self.log_dir / f"{category.value}.log"
                file_handler = logging.handlers.RotatingFileHandler(
                    file_path,
                    maxBytes=self.max_file_size_mb * 1024 * 1024,
                    backupCount=self.backup_count
                )
                file_handler.setFormatter(detailed_formatter)
                file_handler.setLevel(self.level.value)
                
                if self.compress_backups:
                    file_handler.namer = self._compress_namer
                    file_handler.rotator = self._compress_rotator
                
                logger.addHandler(file_handler)
            
            # JSON file handler
            if self.enable_json:
                json_path = self.log_dir / f"{category.value}.json.log"
                json_handler = logging.handlers.RotatingFileHandler(
                    json_path,
                    maxBytes=self.max_file_size_mb * 1024 * 1024,
                    backupCount=self.backup_count
                )
                json_handler.setFormatter(json_formatter)
                json_handler.setLevel(self.level.value)
                logger.addHandler(json_handler)
            
            self._loggers[category] = logger
    
    def _compress_namer(self, name: str) -> str:
        """Name compressed log files"""
        return name + ".gz"
    
    def _compress_rotator(self, source: str, dest: str):
        """Compress rotated log files"""
        with open(source, 'rb') as f_in:
            with gzip.open(dest, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(source)
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for request tracing"""
        self._correlation_id.value = correlation_id
    
    def get_correlation_id(self) -> str:
        """Get current correlation ID"""
        return getattr(self._correlation_id, 'value', '')
    
    def _log(
        self,
        level: LogLevel,
        category: LogCategory,
        message: str,
        data: Optional[Dict] = None,
        trade_id: Optional[str] = None,
        symbol: Optional[str] = None,
        exc_info: bool = False
    ):
        """Internal log method"""
        logger = self._loggers.get(category)
        if not logger:
            return
        
        # Build log entry
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            category=category,
            message=message,
            data=data or {},
            source=self.name,
            correlation_id=self.get_correlation_id(),
            trade_id=trade_id,
            symbol=symbol
        )
        
        # Add to buffer
        with self._buffer_lock:
            self._buffer.append(entry)
        
        # Update counts
        self._log_counts[f"{level.name}_{category.value}"] += 1
        
        # Build log message with context
        context_parts = []
        if entry.correlation_id:
            context_parts.append(f"corr={entry.correlation_id}")
        if trade_id:
            context_parts.append(f"trade={trade_id}")
        if symbol:
            context_parts.append(f"symbol={symbol}")
        
        context_str = f" [{', '.join(context_parts)}]" if context_parts else ""
        
        # Log with data
        if data:
            full_message = f"{message}{context_str} | data={json.dumps(data)}"
        else:
            full_message = f"{message}{context_str}"
        
        # Log to appropriate level
        log_method = getattr(logger, level.name.lower())
        log_method(full_message, exc_info=exc_info)
    
    # Convenience methods
    def debug(self, category: LogCategory, message: str, **kwargs):
        self._log(LogLevel.DEBUG, category, message, **kwargs)
    
    def info(self, category: LogCategory, message: str, **kwargs):
        self._log(LogLevel.INFO, category, message, **kwargs)
    
    def warning(self, category: LogCategory, message: str, **kwargs):
        self._log(LogLevel.WARNING, category, message, **kwargs)
    
    def error(self, category: LogCategory, message: str, exc_info: bool = True, **kwargs):
        self._log(LogLevel.ERROR, category, message, exc_info=exc_info, **kwargs)
    
    def critical(self, category: LogCategory, message: str, exc_info: bool = True, **kwargs):
        self._log(LogLevel.CRITICAL, category, message, exc_info=exc_info, **kwargs)
    
    # Category-specific methods
    def log_trade(self, message: str, trade_id: str, symbol: str, data: Dict = None):
        """Log trade event"""
        self.info(LogCategory.TRADE, message, trade_id=trade_id, symbol=symbol, data=data)
    
    def log_signal(self, message: str, symbol: str, data: Dict = None):
        """Log signal event"""
        self.info(LogCategory.SIGNAL, message, symbol=symbol, data=data)
    
    def log_order(self, message: str, trade_id: str, symbol: str, data: Dict = None):
        """Log order event"""
        self.info(LogCategory.ORDER, message, trade_id=trade_id, symbol=symbol, data=data)
    
    def log_risk(self, message: str, data: Dict = None):
        """Log risk event"""
        self.warning(LogCategory.RISK, message, data=data)
    
    def log_system(self, message: str, data: Dict = None):
        """Log system event"""
        self.info(LogCategory.SYSTEM, message, data=data)
    
    def log_health(self, message: str, data: Dict = None):
        """Log health check"""
        self.info(LogCategory.HEALTH, message, data=data)
    
    def log_performance(self, message: str, data: Dict = None):
        """Log performance metrics"""
        self.info(LogCategory.PERFORMANCE, message, data=data)
    
    def log_audit(self, message: str, data: Dict = None):
        """Log audit event"""
        self.info(LogCategory.AUDIT, message, data=data)
    
    def log_exception(self, category: LogCategory, message: str, exception: Exception):
        """Log exception with traceback"""
        self.error(
            category,
            message,
            data={
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'traceback': traceback.format_exc()
            },
            exc_info=True
        )
    
    def get_recent_logs(self, count: int = 100, category: Optional[LogCategory] = None) -> List[LogEntry]:
        """Get recent log entries"""
        with self._buffer_lock:
            logs = list(self._buffer)
        
        if category:
            logs = [l for l in logs if l.category == category]
        
        return logs[-count:]
    
    def get_log_counts(self) -> Dict[str, int]:
        """Get log counts by level and category"""
        return dict(self._log_counts)
    
    def export_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        category: Optional[LogCategory] = None,
        format: str = "json"
    ) -> str:
        """Export logs to specified format"""
        logs = self.get_recent_logs(10000, category)
        
        if start_time:
            logs = [l for l in logs if l.timestamp >= start_time]
        if end_time:
            logs = [l for l in logs if l.timestamp <= end_time]
        
        if format == "json":
            return json.dumps([l.to_dict() for l in logs], indent=2)
        elif format == "csv":
            output = io.StringIO()
            if logs:
                writer = csv.DictWriter(output, fieldnames=logs[0].to_dict().keys())
                writer.writeheader()
                for log in logs:
                    writer.writerow(log.to_dict())
            return output.getvalue()
        else:
            return "\n".join(l.to_json() for l in logs)


class JsonFormatter(logging.Formatter):
    """JSON log formatter"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


# =============================================================================
# TRADE JOURNAL
# =============================================================================

class TradeJournal:
    """
    Automatic trade documentation and journaling system.
    Records all trade details for analysis and improvement.
    """
    
    def __init__(
        self,
        journal_dir: str = "trade_journal",
        auto_analyze: bool = True
    ):
        self.journal_dir = Path(journal_dir)
        self.auto_analyze = auto_analyze
        
        # Create directories
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        (self.journal_dir / "entries").mkdir(exist_ok=True)
        (self.journal_dir / "reports").mkdir(exist_ok=True)
        (self.journal_dir / "screenshots").mkdir(exist_ok=True)
        
        # In-memory storage
        self._entries: Dict[str, TradeJournalEntry] = {}
        self._lock = threading.Lock()
        
        # Load existing entries
        self._load_entries()
    
    def _load_entries(self):
        """Load existing journal entries"""
        entries_dir = self.journal_dir / "entries"
        for file_path in entries_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    entry = self._dict_to_entry(data)
                    self._entries[entry.trade_id] = entry
            except Exception as e:
                print(f"Error loading journal entry {file_path}: {e}")
    
    def _dict_to_entry(self, data: Dict) -> TradeJournalEntry:
        """Convert dictionary to TradeJournalEntry"""
        # Handle datetime conversion
        if 'entry_time' in data and isinstance(data['entry_time'], str):
            data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        if 'exit_time' in data and isinstance(data['exit_time'], str):
            data['exit_time'] = datetime.fromisoformat(data['exit_time']) if data['exit_time'] else None
        
        return TradeJournalEntry(**data)
    
    def _entry_to_dict(self, entry: TradeJournalEntry) -> Dict:
        """Convert TradeJournalEntry to dictionary"""
        data = asdict(entry)
        data['entry_time'] = entry.entry_time.isoformat()
        data['exit_time'] = entry.exit_time.isoformat() if entry.exit_time else None
        return data
    
    def create_entry(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_time: datetime,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        quantity: float,
        model_name: str = "",
        confluence_factors: List[str] = None,
        session: str = "",
        kill_zone: str = "",
        pre_trade_notes: str = ""
    ) -> TradeJournalEntry:
        """Create new journal entry when trade opens"""
        entry = TradeJournalEntry(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_time=entry_time,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            risk_amount=abs(entry_price - stop_loss) * quantity * 100000,
            model_name=model_name,
            confluence_factors=confluence_factors or [],
            session=session,
            kill_zone=kill_zone,
            pre_trade_notes=pre_trade_notes
        )
        
        with self._lock:
            self._entries[trade_id] = entry
            self._save_entry(entry)
        
        return entry
    
    def close_entry(
        self,
        trade_id: str,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
        pnl: float = None,
        post_trade_notes: str = ""
    ) -> Optional[TradeJournalEntry]:
        """Close journal entry when trade closes"""
        with self._lock:
            entry = self._entries.get(trade_id)
            if not entry:
                return None
            
            entry.exit_time = exit_time
            entry.exit_price = exit_price
            entry.exit_reason = exit_reason
            entry.post_trade_notes = post_trade_notes
            
            # Calculate duration
            if entry.entry_time:
                entry.duration_minutes = int((exit_time - entry.entry_time).total_seconds() / 60)
            
            # Calculate P&L if not provided
            if pnl is not None:
                entry.pnl = pnl
            else:
                if entry.direction.upper() in ['LONG', 'BUY']:
                    entry.pnl = (exit_price - entry.entry_price) * entry.quantity * 100000
                else:
                    entry.pnl = (entry.entry_price - exit_price) * entry.quantity * 100000
            
            # Calculate pips
            pip_size = 0.0001 if entry.entry_price < 10 else 0.01
            if entry.direction.upper() in ['LONG', 'BUY']:
                entry.pnl_pips = (exit_price - entry.entry_price) / pip_size
            else:
                entry.pnl_pips = (entry.entry_price - exit_price) / pip_size
            
            # Calculate R-multiple
            initial_risk = abs(entry.entry_price - entry.stop_loss)
            if initial_risk > 0:
                if entry.direction.upper() in ['LONG', 'BUY']:
                    entry.r_multiple = (exit_price - entry.entry_price) / initial_risk
                else:
                    entry.r_multiple = (entry.entry_price - exit_price) / initial_risk
            
            entry.is_winner = entry.pnl > 0
            
            # Calculate efficiency
            if entry.max_favorable_excursion > 0:
                entry.efficiency_ratio = entry.pnl / entry.max_favorable_excursion
            
            # Auto-generate lessons if enabled
            if self.auto_analyze:
                entry.lessons_learned = self._generate_lessons(entry)
            
            self._save_entry(entry)
            
            return entry
    
    def update_excursions(
        self,
        trade_id: str,
        current_price: float
    ):
        """Update MFE/MAE during trade"""
        with self._lock:
            entry = self._entries.get(trade_id)
            if not entry or entry.exit_time:
                return
            
            if entry.direction.upper() in ['LONG', 'BUY']:
                favorable = (current_price - entry.entry_price) * entry.quantity * 100000
                adverse = (entry.entry_price - current_price) * entry.quantity * 100000
            else:
                favorable = (entry.entry_price - current_price) * entry.quantity * 100000
                adverse = (current_price - entry.entry_price) * entry.quantity * 100000
            
            entry.max_favorable_excursion = max(entry.max_favorable_excursion, max(0, favorable))
            entry.max_adverse_excursion = max(entry.max_adverse_excursion, max(0, adverse))
    
    def _generate_lessons(self, entry: TradeJournalEntry) -> str:
        """Auto-generate lessons learned from trade"""
        lessons = []
        
        # Win/Loss analysis
        if entry.is_winner:
            if entry.r_multiple >= 2:
                lessons.append("✓ Good R:R achieved - patience paid off")
            if entry.efficiency_ratio and entry.efficiency_ratio < 0.5:
                lessons.append("△ Left significant profit on table - consider trailing stop adjustment")
        else:
            if entry.exit_reason == "STOP_LOSS":
                if entry.max_favorable_excursion > entry.risk_amount:
                    lessons.append("✗ Was in profit but gave it back - consider breakeven stop earlier")
                else:
                    lessons.append("✗ Stop loss hit - review entry timing")
            if entry.r_multiple < -1.5:
                lessons.append("✗ Large loss - ensure stops are respected")
        
        # Confluence analysis
        if len(entry.confluence_factors) < 3 and not entry.is_winner:
            lessons.append("△ Low confluence trade - wait for more factors to align")
        
        # Session analysis
        if entry.kill_zone and entry.is_winner:
            lessons.append(f"✓ {entry.kill_zone} kill zone trade worked well")
        elif not entry.kill_zone and not entry.is_winner:
            lessons.append("△ Trade outside kill zone - stick to optimal times")
        
        # Duration analysis
        if entry.duration_minutes and entry.duration_minutes < 5 and entry.is_winner:
            lessons.append("✓ Quick profitable trade - good entry timing")
        elif entry.duration_minutes and entry.duration_minutes > 240 and not entry.is_winner:
            lessons.append("△ Long duration losing trade - consider time-based exits")
        
        return "\n".join(lessons) if lessons else "No specific lessons identified"
    
    def _save_entry(self, entry: TradeJournalEntry):
        """Save entry to file"""
        file_path = self.journal_dir / "entries" / f"{entry.trade_id}.json"
        with open(file_path, 'w') as f:
            json.dump(self._entry_to_dict(entry), f, indent=2)
    
    def get_entry(self, trade_id: str) -> Optional[TradeJournalEntry]:
        """Get journal entry by trade ID"""
        return self._entries.get(trade_id)
    
    def get_all_entries(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbol: Optional[str] = None,
        model: Optional[str] = None,
        winners_only: bool = False,
        losers_only: bool = False
    ) -> List[TradeJournalEntry]:
        """Get filtered journal entries"""
        entries = list(self._entries.values())
        
        if start_date:
            entries = [e for e in entries if e.entry_time >= start_date]
        if end_date:
            entries = [e for e in entries if e.entry_time <= end_date]
        if symbol:
            entries = [e for e in entries if e.symbol == symbol]
        if model:
            entries = [e for e in entries if e.model_name == model]
        if winners_only:
            entries = [e for e in entries if e.is_winner]
        if losers_only:
            entries = [e for e in entries if e.exit_time and not e.is_winner]
        
        return sorted(entries, key=lambda e: e.entry_time, reverse=True)
    
    def generate_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        format: str = "markdown"
    ) -> str:
        """Generate trading report"""
        entries = self.get_all_entries(start_date, end_date)
        closed_entries = [e for e in entries if e.exit_time]
        
        if not closed_entries:
            return "No closed trades in period"
        
        # Calculate statistics
        total_trades = len(closed_entries)
        winners = [e for e in closed_entries if e.is_winner]
        losers = [e for e in closed_entries if not e.is_winner]
        
        win_rate = len(winners) / total_trades if total_trades > 0 else 0
        total_pnl = sum(e.pnl for e in closed_entries)
        avg_winner = sum(e.pnl for e in winners) / len(winners) if winners else 0
        avg_loser = sum(e.pnl for e in losers) / len(losers) if losers else 0
        avg_r = sum(e.r_multiple for e in closed_entries) / total_trades if total_trades > 0 else 0
        
        # Model performance
        model_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0})
        for e in closed_entries:
            model_stats[e.model_name]['trades'] += 1
            if e.is_winner:
                model_stats[e.model_name]['wins'] += 1
            model_stats[e.model_name]['pnl'] += e.pnl
        
        # Session performance
        session_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0})
        for e in closed_entries:
            session_stats[e.session]['trades'] += 1
            if e.is_winner:
                session_stats[e.session]['wins'] += 1
            session_stats[e.session]['pnl'] += e.pnl
        
        if format == "markdown":
            return self._generate_markdown_report(
                start_date, end_date, total_trades, win_rate, total_pnl,
                avg_winner, avg_loser, avg_r, model_stats, session_stats, closed_entries
            )
        else:
            return self._generate_html_report(
                start_date, end_date, total_trades, win_rate, total_pnl,
                avg_winner, avg_loser, avg_r, model_stats, session_stats, closed_entries
            )
    
    def _generate_markdown_report(
        self, start_date, end_date, total_trades, win_rate, total_pnl,
        avg_winner, avg_loser, avg_r, model_stats, session_stats, entries
    ) -> str:
        """Generate Markdown report"""
        lines = [
            "# ICT Trading Journal Report",
            "",
            f"**Period:** {start_date.strftime('%Y-%m-%d') if start_date else 'All Time'} to {end_date.strftime('%Y-%m-%d') if end_date else 'Present'}",
            f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "",
            "## Summary Statistics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Trades | {total_trades} |",
            f"| Win Rate | {win_rate:.1%} |",
            f"| Total P&L | ${total_pnl:,.2f} |",
            f"| Avg Winner | ${avg_winner:,.2f} |",
            f"| Avg Loser | ${avg_loser:,.2f} |",
            f"| Avg R-Multiple | {avg_r:.2f}R |",
            "",
            "## Performance by Model",
            "",
            "| Model | Trades | Win Rate | P&L |",
            "|-------|--------|----------|-----|"
        ]
        
        for model, stats in model_stats.items():
            wr = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
            lines.append(f"| {model or 'Unknown'} | {stats['trades']} | {wr:.1%} | ${stats['pnl']:,.2f} |")
        
        lines.extend([
            "",
            "## Performance by Session",
            "",
            "| Session | Trades | Win Rate | P&L |",
            "|---------|--------|----------|-----|"
        ])
        
        for session, stats in session_stats.items():
            wr = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
            lines.append(f"| {session or 'Unknown'} | {stats['trades']} | {wr:.1%} | ${stats['pnl']:,.2f} |")
        
        lines.extend([
            "",
            "## Recent Trades",
            "",
            "| Date | Symbol | Direction | P&L | R | Model |",
            "|------|--------|-----------|-----|---|-------|"
        ])
        
        for entry in entries[:20]:
            date_str = entry.entry_time.strftime('%Y-%m-%d %H:%M')
            pnl_str = f"${entry.pnl:+,.2f}" if entry.pnl else "Open"
            lines.append(f"| {date_str} | {entry.symbol} | {entry.direction} | {pnl_str} | {entry.r_multiple:.1f}R | {entry.model_name} |")
        
        lines.extend([
            "",
            "## Lessons Learned",
            ""
        ])
        
        for entry in entries[:10]:
            if entry.lessons_learned:
                lines.append(f"**{entry.trade_id}** ({entry.symbol} {entry.direction}):")
                lines.append(entry.lessons_learned)
                lines.append("")
        
        return "\n".join(lines)
    
    def _generate_html_report(
        self, start_date, end_date, total_trades, win_rate, total_pnl,
        avg_winner, avg_loser, avg_r, model_stats, session_stats, entries
    ) -> str:
        """Generate HTML report"""
        # Simplified HTML report
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>ICT Trading Journal Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #667eea; color: white; }}
        .positive {{ color: #4ade80; }}
        .negative {{ color: #f87171; }}
        .metric-card {{ display: inline-block; padding: 20px; margin: 10px; background: #f8f9fa; border-radius: 8px; min-width: 150px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 ICT Trading Journal Report</h1>
        <p>Period: {start_date.strftime('%Y-%m-%d') if start_date else 'All Time'} to {end_date.strftime('%Y-%m-%d') if end_date else 'Present'}</p>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{win_rate:.1%}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if total_pnl >= 0 else 'negative'}">${total_pnl:,.2f}</div>
                <div class="metric-label">Total P&L</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{avg_r:.2f}R</div>
                <div class="metric-label">Avg R-Multiple</div>
            </div>
        </div>
        
        <h2>Recent Trades</h2>
        <table>
            <thead>
                <tr><th>Date</th><th>Symbol</th><th>Direction</th><th>P&L</th><th>R</th><th>Model</th></tr>
            </thead>
            <tbody>
                {''.join(f"<tr><td>{e.entry_time.strftime('%Y-%m-%d %H:%M')}</td><td>{e.symbol}</td><td>{e.direction}</td><td class='{'positive' if e.pnl >= 0 else 'negative'}'>${e.pnl:+,.2f}</td><td>{e.r_multiple:.1f}R</td><td>{e.model_name}</td></tr>" for e in entries[:20])}
            </tbody>
        </table>
    </div>
</body>
</html>
        """
    
    def export_csv(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> str:
        """Export entries to CSV"""
        entries = self.get_all_entries(start_date, end_date)
        
        output = io.StringIO()
        if entries:
            fieldnames = [
                'trade_id', 'symbol', 'direction', 'entry_time', 'exit_time',
                'entry_price', 'exit_price', 'stop_loss', 'take_profit',
                'quantity', 'pnl', 'pnl_pips', 'r_multiple', 'is_winner',
                'exit_reason', 'model_name', 'session', 'kill_zone'
            ]
            
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in entries:
                row = {
                    'trade_id': entry.trade_id,
                    'symbol': entry.symbol,
                    'direction': entry.direction,
                    'entry_time': entry.entry_time.isoformat(),
                    'exit_time': entry.exit_time.isoformat() if entry.exit_time else '',
                    'entry_price': entry.entry_price,
                    'exit_price': entry.exit_price,
                    'stop_loss': entry.stop_loss,
                    'take_profit': entry.take_profit,
                    'quantity': entry.quantity,
                    'pnl': entry.pnl,
                    'pnl_pips': entry.pnl_pips,
                    'r_multiple': entry.r_multiple,
                    'is_winner': entry.is_winner,
                    'exit_reason': entry.exit_reason,
                    'model_name': entry.model_name,
                    'session': entry.session,
                    'kill_zone': entry.kill_zone
                }
                writer.writerow(row)
        
        return output.getvalue()


# =============================================================================
# HEALTH MONITORING
# =============================================================================

class HealthMonitor:
    """
    System health monitoring and alerting.
    Monitors connections, resources, and API limits.
    """
    
    def __init__(
        self,
        check_interval_seconds: int = 60,
        alert_callback: Optional[Callable] = None
    ):
        self.check_interval = check_interval_seconds
        self.alert_callback = alert_callback
        
        self._checks: Dict[str, Callable] = {}
        self._results: Dict[str, HealthCheckResult] = {}
        self._history: deque = deque(maxlen=1000)
        self._rate_limits: Dict[str, APIRateLimitStatus] = {}
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Register default checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks"""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("network", self._check_network)
    
    def register_check(self, name: str, check_func: Callable):
        """Register health check function"""
        self._checks[name] = check_func
    
    def register_broker_check(self, broker):
        """Register broker connection check"""
        def check_broker() -> HealthCheckResult:
            start = time.time()
            try:
                connected = broker.is_connected()
                latency = (time.time() - start) * 1000
                
                if connected:
                    return HealthCheckResult(
                        name="broker_connection",
                        status=HealthStatus.HEALTHY,
                        message="Broker connected",
                        latency_ms=latency
                    )
                else:
                    return HealthCheckResult(
                        name="broker_connection",
                        status=HealthStatus.UNHEALTHY,
                        message="Broker disconnected",
                        latency_ms=latency
                    )
            except Exception as e:
                return HealthCheckResult(
                    name="broker_connection",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Broker check failed: {str(e)}"
                )
        
        self.register_check("broker_connection", check_broker)
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check system resources"""
        try:
            import os
            
            # CPU (simplified - would use psutil in production)
            load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
            cpu_count = os.cpu_count() or 1
            cpu_percent = (load_avg / cpu_count) * 100
            
            # Memory (simplified)
            # In production, use psutil.virtual_memory()
            memory_percent = 50  # Placeholder
            
            details = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'load_average': load_avg
            }
            
            if cpu_percent > 90 or memory_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "High resource usage"
            elif cpu_percent > 70 or memory_percent > 70:
                status = HealthStatus.DEGRADED
                message = "Elevated resource usage"
            else:
                status = HealthStatus.HEALTHY
                message = "Resources normal"
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message=f"Check failed: {str(e)}"
            )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space"""
        try:
            import shutil
            
            total, used, free = shutil.disk_usage("/")
            
            free_percent = (free / total) * 100
            used_gb = used / (1024**3)
            free_gb = free / (1024**3)
            
            details = {
                'free_percent': free_percent,
                'used_gb': used_gb,
                'free_gb': free_gb
            }
            
            if free_percent < 5:
                status = HealthStatus.UNHEALTHY
                message = f"Critical: Only {free_gb:.1f}GB free"
            elif free_percent < 15:
                status = HealthStatus.DEGRADED
                message = f"Low disk space: {free_gb:.1f}GB free"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk OK: {free_gb:.1f}GB free"
            
            return HealthCheckResult(
                name="disk_space",
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Check failed: {str(e)}"
            )
    
    def _check_network(self) -> HealthCheckResult:
        """Check network connectivity"""
        try:
            start = time.time()
            
            # Try to connect to common endpoints
            test_hosts = [
                ("8.8.8.8", 53),  # Google DNS
                ("1.1.1.1", 53),  # Cloudflare DNS
            ]
            
            connected = False
            for host, port in test_hosts:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    if result == 0:
                        connected = True
                        break
                except:
                    continue
            
            latency = (time.time() - start) * 1000
            
            if connected:
                if latency > 1000:
                    status = HealthStatus.DEGRADED
                    message = f"Network slow: {latency:.0f}ms"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Network OK: {latency:.0f}ms"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Network unreachable"
            
            return HealthCheckResult(
                name="network",
                status=status,
                message=message,
                latency_ms=latency
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="network",
                status=HealthStatus.UNKNOWN,
                message=f"Check failed: {str(e)}"
            )
    
    def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run single health check"""
        check_func = self._checks.get(name)
        if not check_func:
            return None
        
        try:
            result = check_func()
            
            with self._lock:
                self._results[name] = result
                self._history.append(result)
            
            # Alert on unhealthy
            if result.status == HealthStatus.UNHEALTHY and self.alert_callback:
                self.alert_callback(result)
            
            return result
            
        except Exception as e:
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Check exception: {str(e)}"
            )
            return result
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}
        
        for name in self._checks:
            result = self.run_check(name)
            if result:
                results[name] = result
        
        return results
    
    def update_rate_limit(
        self,
        api_name: str,
        requests_made: int,
        requests_limit: int,
        reset_time: datetime
    ):
        """Update API rate limit status"""
        status = APIRateLimitStatus(
            api_name=api_name,
            requests_made=requests_made,
            requests_limit=requests_limit,
            requests_remaining=requests_limit - requests_made,
            reset_time=reset_time,
            is_limited=requests_made >= requests_limit
        )
        
        with self._lock:
            self._rate_limits[api_name] = status
        
        # Alert if approaching limit
        if status.usage_percent > 80 and self.alert_callback:
            self.alert_callback(HealthCheckResult(
                name=f"rate_limit_{api_name}",
                status=HealthStatus.DEGRADED,
                message=f"API rate limit at {status.usage_percent:.0f}%",
                details={'rate_limit': asdict(status)}
            ))
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status"""
        if not self._results:
            return HealthStatus.UNKNOWN
        
        statuses = [r.status for r in self._results.values()]
        
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_status_report(self) -> Dict:
        """Get complete status report"""
        return {
            'overall_status': self.get_overall_status().value,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {name: result.to_dict() for name, result in self._results.items()},
            'rate_limits': {name: asdict(status) for name, status in self._rate_limits.items()}
        }
    
    def start(self):
        """Start background monitoring"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop background monitoring"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._running:
            self.run_all_checks()
            time.sleep(self.check_interval)


# =============================================================================
# DEPLOYMENT CONFIGURATIONS
# =============================================================================

class DeploymentConfig:
    """Deployment configuration generator"""
    
    @staticmethod
    def generate_dockerfile() -> str:
        """Generate Dockerfile for containerization"""
        return '''# ICT Trading Bot Dockerfile
# Multi-stage build for smaller final image

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    libffi-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash trader
WORKDIR /home/trader/app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/trader/.local

# Copy application code
COPY --chown=trader:trader . .

# Set environment
ENV PATH=/home/trader/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create necessary directories
RUN mkdir -p logs trade_journal data && chown -R trader:trader .

# Switch to non-root user
USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Expose ports
EXPOSE 5000

# Default command
CMD ["python", "live_trader.py"]
'''
    
    @staticmethod
    def generate_docker_compose() -> str:
        """Generate docker-compose.yml"""
        return '''version: '3.8'

services:
  trading-bot:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ict-trading-bot
    restart: unless-stopped
    ports:
      - "5000:5000"
    volumes:
      - ./logs:/home/trader/app/logs
      - ./trade_journal:/home/trader/app/trade_journal
      - ./data:/home/trader/app/data
      - ./config:/home/trader/app/config:ro
    environment:
      - TRADING_MODE=paper
      - LOG_LEVEL=INFO
      - DASHBOARD_ENABLED=true
      - TZ=America/New_York
    env_file:
      - .env
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "5"
    networks:
      - trading-network

  # Optional: Redis for caching/queuing
  redis:
    image: redis:7-alpine
    container_name: ict-redis
    restart: unless-stopped
    volumes:
      - redis-data:/data
    networks:
      - trading-network

  # Optional: Prometheus for metrics
  prometheus:
    image: prom/prometheus:latest
    container_name: ict-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    networks:
      - trading-network

  # Optional: Grafana for dashboards
  grafana:
    image: grafana/grafana:latest
    container_name: ict-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    networks:
      - trading-network

networks:
  trading-network:
    driver: bridge

volumes:
  redis-data:
  prometheus-data:
  grafana-data:
'''
    
    @staticmethod
    def generate_requirements() -> str:
        """Generate requirements.txt"""
        return '''# ICT Trading Bot Requirements

# Core
python-dateutil>=2.8.2
pytz>=2023.3

# Data processing
numpy>=1.24.0
pandas>=2.0.0

# Machine Learning (optional)
scikit-learn>=1.3.0
# tensorflow>=2.13.0  # Uncomment for LSTM
# torch>=2.0.0  # Uncomment for PyTorch

# Broker APIs
requests>=2.31.0
websocket-client>=1.6.0
# MetaTrader5>=5.0.45  # Windows only

# Web Dashboard
flask>=2.3.0
# streamlit>=1.25.0  # Alternative dashboard

# Monitoring
prometheus-client>=0.17.0
psutil>=5.9.0

# Notifications
python-telegram-bot>=20.0
# discord.py>=2.3.0  # For Discord alerts

# Database (optional)
# sqlalchemy>=2.0.0
# psycopg2-binary>=2.9.0  # PostgreSQL
# redis>=4.6.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0

# Code Quality
black>=23.0.0
flake8>=6.1.0
mypy>=1.5.0
'''
    
    @staticmethod
    def generate_env_template() -> str:
        """Generate .env.template"""
        return '''# ICT Trading Bot Environment Configuration
# Copy this file to .env and fill in your values

# ===================
# Trading Configuration
# ===================
TRADING_MODE=paper  # paper, live, shadow
SYMBOLS=EUR/USD,GBP/USD
TIMEFRAMES=M15,H1,H4
RISK_PER_TRADE=1.0
MAX_DAILY_TRADES=10
MAX_DAILY_LOSS_PCT=3.0

# ===================
# Broker Configuration
# ===================
# OANDA
OANDA_API_KEY=your_api_key_here
OANDA_ACCOUNT_ID=your_account_id
OANDA_ENVIRONMENT=practice  # practice or live

# MT5 (Windows only)
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server

# ===================
# Alert Configuration
# ===================
# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Email
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
EMAIL_FROM=your_email@gmail.com
EMAIL_TO=recipient@example.com

# Discord
DISCORD_WEBHOOK_URL=your_webhook_url

# ===================
# Dashboard Configuration
# ===================
DASHBOARD_ENABLED=true
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=5000
DASHBOARD_SECRET_KEY=generate_a_secure_random_key

# ===================
# Logging Configuration
# ===================
LOG_LEVEL=INFO
LOG_DIR=logs
ENABLE_JSON_LOGS=true

# ===================
# Database Configuration (Optional)
# ===================
# DATABASE_URL=postgresql://user:pass@localhost:5432/trading
# REDIS_URL=redis://localhost:6379/0

# ===================
# Cloud Configuration (AWS)
# ===================
# AWS_ACCESS_KEY_ID=your_access_key
# AWS_SECRET_ACCESS_KEY=your_secret_key
# AWS_REGION=us-east-1
# S3_BUCKET=your-trading-bucket

# ===================
# Cloud Configuration (GCP)
# ===================
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
# GCP_PROJECT_ID=your-project-id
# GCS_BUCKET=your-trading-bucket
'''
    
    @staticmethod
    def generate_aws_cloudformation() -> str:
        """Generate AWS CloudFormation template"""
        return '''AWSTemplateFormatVersion: '2010-09-09'
Description: ICT Trading Bot Infrastructure

Parameters:
  Environment:
    Type: String
    Default: staging
    AllowedValues: [staging, production]
  InstanceType:
    Type: String
    Default: t3.small
    AllowedValues: [t3.micro, t3.small, t3.medium, t3.large]

Resources:
  # VPC
  TradingVPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub ${Environment}-trading-vpc

  # Subnet
  PublicSubnet:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref TradingVPC
      CidrBlock: 10.0.1.0/24
      MapPublicIpOnLaunch: true
      AvailabilityZone: !Select [0, !GetAZs '']
      Tags:
        - Key: Name
          Value: !Sub ${Environment}-public-subnet

  # Internet Gateway
  InternetGateway:
    Type: AWS::EC2::InternetGateway
    Properties:
      Tags:
        - Key: Name
          Value: !Sub ${Environment}-igw

  AttachGateway:
    Type: AWS::EC2::VPCGatewayAttachment
    Properties:
      VpcId: !Ref TradingVPC
      InternetGatewayId: !Ref InternetGateway

  # Route Table
  PublicRouteTable:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref TradingVPC

  PublicRoute:
    Type: AWS::EC2::Route
    DependsOn: AttachGateway
    Properties:
      RouteTableId: !Ref PublicRouteTable
      DestinationCidrBlock: 0.0.0.0/0
      GatewayId: !Ref InternetGateway

  SubnetRouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      SubnetId: !Ref PublicSubnet
      RouteTableId: !Ref PublicRouteTable

  # Security Group
  TradingSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Trading Bot Security Group
      VpcId: !Ref TradingVPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          CidrIp: 0.0.0.0/0  # Restrict in production!
        - IpProtocol: tcp
          FromPort: 5000
          ToPort: 5000
          CidrIp: 0.0.0.0/0
      SecurityGroupEgress:
        - IpProtocol: -1
          CidrIp: 0.0.0.0/0

  # EC2 Instance
  TradingInstance:
    Type: AWS::EC2::Instance
    Properties:
      InstanceType: !Ref InstanceType
      ImageId: ami-0c55b159cbfafe1f0  # Amazon Linux 2 (update for your region)
      SubnetId: !Ref PublicSubnet
      SecurityGroupIds:
        - !Ref TradingSecurityGroup
      IamInstanceProfile: !Ref TradingInstanceProfile
      UserData:
        Fn::Base64: !Sub |
          #!/bin/bash
          yum update -y
          yum install -y docker
          service docker start
          usermod -a -G docker ec2-user
          
          # Install docker-compose
          curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
          chmod +x /usr/local/bin/docker-compose
          
          # Clone and start application
          # git clone your-repo /home/ec2-user/trading-bot
          # cd /home/ec2-user/trading-bot
          # docker-compose up -d
      Tags:
        - Key: Name
          Value: !Sub ${Environment}-trading-bot

  # IAM Role
  TradingInstanceRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: ec2.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
        - arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy

  TradingInstanceProfile:
    Type: AWS::IAM::InstanceProfile
    Properties:
      Roles:
        - !Ref TradingInstanceRole

  # CloudWatch Log Group
  TradingLogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub /trading-bot/${Environment}
      RetentionInDays: 30

  # CloudWatch Alarms
  HighCPUAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmDescription: Alert when CPU > 80%
      MetricName: CPUUtilization
      Namespace: AWS/EC2
      Statistic: Average
      Period: 300
      EvaluationPeriods: 2
      Threshold: 80
      ComparisonOperator: GreaterThanThreshold
      Dimensions:
        - Name: InstanceId
          Value: !Ref TradingInstance

Outputs:
  InstancePublicIP:
    Description: Public IP of trading bot instance
    Value: !GetAtt TradingInstance.PublicIp
  DashboardURL:
    Description: Trading dashboard URL
    Value: !Sub http://${TradingInstance.PublicIp}:5000
'''
    
    @staticmethod
    def generate_gcp_terraform() -> str:
        """Generate GCP Terraform configuration"""
        return '''# ICT Trading Bot - GCP Terraform Configuration

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment (staging/production)"
  type        = string
  default     = "staging"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# VPC Network
resource "google_compute_network" "trading_network" {
  name                    = "${var.environment}-trading-network"
  auto_create_subnetworks = false
}

# Subnet
resource "google_compute_subnetwork" "trading_subnet" {
  name          = "${var.environment}-trading-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.trading_network.id
}

# Firewall - SSH
resource "google_compute_firewall" "ssh" {
  name    = "${var.environment}-allow-ssh"
  network = google_compute_network.trading_network.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]  # Restrict in production!
  target_tags   = ["trading-bot"]
}

# Firewall - Dashboard
resource "google_compute_firewall" "dashboard" {
  name    = "${var.environment}-allow-dashboard"
  network = google_compute_network.trading_network.name

  allow {
    protocol = "tcp"
    ports    = ["5000"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["trading-bot"]
}

# Service Account
resource "google_service_account" "trading_sa" {
  account_id   = "${var.environment}-trading-bot"
  display_name = "Trading Bot Service Account"
}

# Compute Instance
resource "google_compute_instance" "trading_bot" {
  name         = "${var.environment}-trading-bot"
  machine_type = "e2-small"
  zone         = var.zone

  tags = ["trading-bot"]

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
      size  = 20
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.trading_subnet.id
    access_config {
      // Ephemeral public IP
    }
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y docker.io docker-compose
    systemctl start docker
    systemctl enable docker
    
    # Add user to docker group
    usermod -aG docker $USER
    
    # Setup application
    mkdir -p /opt/trading-bot
    cd /opt/trading-bot
    
    # Clone your repository here
    # git clone your-repo .
    # docker-compose up -d
  EOF

  service_account {
    email  = google_service_account.trading_sa.email
    scopes = ["cloud-platform"]
  }

  labels = {
    environment = var.environment
    application = "trading-bot"
  }
}

# Cloud Storage for logs/data
resource "google_storage_bucket" "trading_data" {
  name     = "${var.project_id}-${var.environment}-trading-data"
  location = var.region
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}

# Outputs
output "instance_ip" {
  value = google_compute_instance.trading_bot.network_interface[0].access_config[0].nat_ip
}

output "dashboard_url" {
  value = "http://${google_compute_instance.trading_bot.network_interface[0].access_config[0].nat_ip}:5000"
}

output "storage_bucket" {
  value = google_storage_bucket.trading_data.name
}
'''
    
    @staticmethod
    def generate_github_actions() -> str:
        """Generate GitHub Actions CI/CD workflow"""
        return '''name: ICT Trading Bot CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  DOCKER_IMAGE: ict-trading-bot
  AWS_REGION: us-east-1

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov flake8
      
      - name: Lint with flake8
        run: |
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
      
      - name: Run tests
        run: |
          pytest tests/ --cov=. --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            ${{ secrets.DOCKER_USERNAME }}/${{ env.DOCKER_IMAGE }}:${{ github.sha }}
            ${{ secrets.DOCKER_USERNAME }}/${{ env.DOCKER_IMAGE }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    steps:
      - name: Deploy to staging
        run: |
          echo "Deploying to staging..."
          # Add deployment commands here
          # ssh user@staging-server "cd /app && docker-compose pull && docker-compose up -d"

  deploy-production:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
      - name: Deploy to production
        run: |
          echo "Deploying to production..."
          # Add deployment commands here
'''
    
    @staticmethod
    def save_all_configs(output_dir: str = "deployment"):
        """Save all deployment configuration files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        configs = {
            'Dockerfile': DeploymentConfig.generate_dockerfile(),
            'docker-compose.yml': DeploymentConfig.generate_docker_compose(),
            'requirements.txt': DeploymentConfig.generate_requirements(),
            '.env.template': DeploymentConfig.generate_env_template(),
            'aws-cloudformation.yml': DeploymentConfig.generate_aws_cloudformation(),
            'gcp-terraform.tf': DeploymentConfig.generate_gcp_terraform(),
            '.github/workflows/ci-cd.yml': DeploymentConfig.generate_github_actions(),
        }
        
        for filename, content in configs.items():
            file_path = output_path / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Created: {file_path}")
        
        return output_path


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class ICTMonitoringSystem:
    """
    Complete monitoring system integrating all components.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        journal_dir: str = "trade_journal",
        environment: DeploymentEnvironment = DeploymentEnvironment.DEVELOPMENT
    ):
        self.environment = environment
        
        # Initialize components
        self.logger = StructuredLogger(
            log_dir=log_dir,
            level=LogLevel.DEBUG if environment == DeploymentEnvironment.DEVELOPMENT else LogLevel.INFO
        )
        
        self.journal = TradeJournal(journal_dir=journal_dir)
        
        self.health_monitor = HealthMonitor(
            check_interval_seconds=60,
            alert_callback=self._handle_health_alert
        )
        
        self.logger.log_system(
            f"ICT Monitoring System initialized",
            data={'environment': environment.value}
        )
    
    def _handle_health_alert(self, result: HealthCheckResult):
        """Handle health check alerts"""
        self.logger.warning(
            LogCategory.HEALTH,
            f"Health alert: {result.name} - {result.message}",
            data=result.to_dict()
        )
    
    def start(self):
        """Start monitoring system"""
        self.health_monitor.start()
        self.logger.log_system("Monitoring system started")
    
    def stop(self):
        """Stop monitoring system"""
        self.health_monitor.stop()
        self.logger.log_system("Monitoring system stopped")
    
    def get_status(self) -> Dict:
        """Get complete system status"""
        return {
            'environment': self.environment.value,
            'health': self.health_monitor.get_status_report(),
            'log_counts': self.logger.get_log_counts(),
            'recent_trades': len(self.journal.get_all_entries()),
            'timestamp': datetime.utcnow().isoformat()
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ICT LOGGING, MONITORING & DEPLOYMENT SYSTEM")
    print("=" * 60)
    
    # Initialize monitoring system
    monitoring = ICTMonitoringSystem(
        log_dir="logs",
        journal_dir="trade_journal",
        environment=DeploymentEnvironment.DEVELOPMENT
    )
    
    # Start monitoring
    monitoring.start()
    
    # Example: Structured logging
    print("\n--- Structured Logging Demo ---")
    monitoring.logger.log_trade(
        "Trade opened",
        trade_id="T001",
        symbol="EUR/USD",
        data={'direction': 'LONG', 'entry': 1.0850}
    )
    
    monitoring.logger.log_signal(
        "Signal generated",
        symbol="GBP/USD",
        data={'model': 'Silver Bullet', 'confidence': 0.85}
    )
    
    monitoring.logger.log_risk(
        "Drawdown warning",
        data={'current_drawdown': 2.5, 'limit': 3.0}
    )
    
    # Example: Trade Journal
    print("\n--- Trade Journal Demo ---")
    entry = monitoring.journal.create_entry(
        trade_id="T001",
        symbol="EUR/USD",
        direction="LONG",
        entry_time=datetime.utcnow(),
        entry_price=1.0850,
        stop_loss=1.0820,
        take_profit=1.0910,
        quantity=0.1,
        model_name="Order Block",
        confluence_factors=["Bullish OB", "FVG", "HTF Trend"],
        session="london",
        kill_zone="London Open"
    )
    print(f"Created journal entry: {entry.trade_id}")
    
    # Simulate trade close
    monitoring.journal.close_entry(
        trade_id="T001",
        exit_time=datetime.utcnow() + timedelta(hours=2),
        exit_price=1.0890,
        exit_reason="TAKE_PROFIT"
    )
    
    # Generate report
    report = monitoring.journal.generate_report()
    print("\n--- Journal Report ---")
    print(report[:500] + "...\n")
    
    # Example: Health Checks
    print("\n--- Health Check Demo ---")
    results = monitoring.health_monitor.run_all_checks()
    for name, result in results.items():
        print(f"  {name}: {result.status.value} - {result.message}")
    
    # Get overall status
    status = monitoring.get_status()
    print(f"\nOverall Health: {status['health']['overall_status']}")
    
    # Example: Generate deployment configs
    print("\n--- Generating Deployment Configs ---")
    DeploymentConfig.save_all_configs("deployment")
    
    # Stop monitoring
    monitoring.stop()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
