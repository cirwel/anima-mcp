"""
Unified Workflow Orchestrator - Coordinates both MCP servers.

Provides unified interface for agents to interact with:
- anima-mcp (Lumen's proprioceptive state)
- unitares-governance (Governance decisions)

Enables cross-server workflows and orchestration.
"""

import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from .anima import Anima, sense_self
from .sensors.base import SensorReadings
from .eisv_mapper import EISVMetrics, anima_to_eisv
from .unitares_bridge import UnitaresBridge
from .shared_memory import SharedMemoryClient
from .config import get_calibration
from .server_state import SHM_STALE_THRESHOLD_SECONDS, anima_from_dict, readings_from_dict


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # Some steps succeeded


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    name: str
    server: str  # "anima" or "unitares"
    tool: str
    arguments: Dict[str, Any]
    depends_on: List[str] = None  # Step names this depends on
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    status: WorkflowStatus
    steps: Dict[str, Any]  # Step name -> result
    errors: Dict[str, str]  # Step name -> error message
    summary: str


class UnifiedWorkflowOrchestrator:
    """
    Orchestrates workflows across both MCP servers.
    
    Provides unified interface for:
    - Checking Lumen's state
    - Getting governance decisions
    - Coordinating multi-step workflows
    """
    
    def __init__(
        self,
        unitares_url: Optional[str] = None,
        anima_store=None,  # IdentityStore
        anima_sensors=None,  # SensorBackend
    ):
        """
        Initialize orchestrator.
        
        Args:
            unitares_url: URL to UNITARES governance server
            anima_store: Lumen's identity store (for local state)
            anima_sensors: Lumen's sensor backend (for local reads)
        """
        self._unitares_url = unitares_url
        self._anima_store = anima_store
        self._anima_sensors = anima_sensors  # Kept for fallback, but prefer shared memory
        self._bridge: Optional[UnitaresBridge] = None
        self._shm_client: Optional[SharedMemoryClient] = None
        
        if unitares_url:
            self._bridge = UnitaresBridge(unitares_url=unitares_url)
    
    def _get_readings_and_anima(self) -> tuple[SensorReadings | None, Anima | None]:
        """
        Read sensor data from shared memory (broker) or fallback to direct sensor access.
        
        Returns:
            Tuple of (readings, anima) or (None, None) if unavailable
        """
        # Try shared memory first (broker mode)
        if self._shm_client is None:
            # Use file backend to match broker (Redis caused hangs)
            self._shm_client = SharedMemoryClient(mode="read", backend="file")
        
        shm_data = self._shm_client.read()
        
        if shm_data and "readings" in shm_data and "anima" in shm_data:
            try:
                # Reconstruct SensorReadings from shared memory
                readings_dict = shm_data["readings"]

                state_timestamp = shm_data.get("timestamp") or readings_dict.get("timestamp")
                if not isinstance(state_timestamp, str) or not state_timestamp:
                    raise ValueError("shared state has no timestamp")
                state_time = datetime.fromisoformat(state_timestamp.replace("Z", "+00:00"))
                age_seconds = (datetime.now(state_time.tzinfo) - state_time).total_seconds()
                if age_seconds > SHM_STALE_THRESHOLD_SECONDS or age_seconds < -5:
                    raise ValueError("shared state is stale or future-dated")
                
                readings = readings_from_dict(readings_dict)
                anima = anima_from_dict(shm_data["anima"], readings)
                
                return readings, anima
            except Exception:
                # Fall through to direct sensor access
                pass
        
        # Fallback to direct sensor access if shared memory unavailable
        if self._anima_sensors:
            try:
                readings = self._anima_sensors.read()
                calibration = get_calibration()
                anima = sense_self(readings, calibration)
                return readings, anima
            except Exception:
                pass
        
        return None, None
    
    async def check_unitares_available(self) -> bool:
        """Check if UNITARES server is available."""
        if not self._bridge:
            return False
        try:
            return await self._bridge.check_availability()
        except Exception:
            return False
    
    async def get_lumen_state(self) -> Dict[str, Any]:
        """
        Get Lumen's current state (local, no MCP call needed).
        
        Returns:
            Dict with anima state, sensors, identity, etc.
        """
        if not self._anima_store:
            return {"error": "Anima store not available"}
        
        # Read from shared memory (broker) or fallback to sensors
        readings, anima = self._get_readings_and_anima()
        if readings is None or anima is None:
            return {"error": "Unable to read sensor data"}
        
        # Get identity
        identity = self._anima_store.get_identity()
        
        # Compute EISV
        eisv = anima_to_eisv(anima, readings)
        
        return {
            "anima": {
                "warmth": anima.warmth,
                "clarity": anima.clarity,
                "stability": anima.stability,
                "presence": anima.presence,
            },
            "eisv": eisv.to_dict(),
            "sensors": readings.to_dict(),
            "identity": {
                "name": identity.name,
                "id": identity.creature_id[:8] + "...",
                "awakenings": identity.total_awakenings,
                "alive_seconds": round(identity.total_alive_seconds),
            },
            "timestamp": readings.timestamp.isoformat() if readings.timestamp else None,
        }
    
    async def check_governance(
        self,
        anima: Optional[Anima] = None,
        readings: Optional[SensorReadings] = None,
        eisv: Optional[EISVMetrics] = None
    ) -> Dict[str, Any]:
        """
        Check governance decision from UNITARES.
        
        Args:
            anima: Anima state (computed if None)
            readings: Sensor readings (read if None)
            eisv: EISV metrics (computed if None)
        
        Returns:
            Governance decision dict
        """
        if not self._bridge:
            return {"error": "UNITARES bridge not configured"}
        
        # Compute missing values - try shared memory first
        if not readings or not anima:
            readings, anima = self._get_readings_and_anima()
        
        if not eisv and anima and readings:
            eisv = anima_to_eisv(anima, readings)
        
        if not anima or not readings or not eisv:
            return {"error": "Cannot compute governance - missing data"}
        
        try:
            result = await self._bridge.check_in(anima, readings)
            return result
        except Exception as e:
            return {"error": str(e), "source": "local"}
    
    async def execute_workflow(
        self,
        steps: List[WorkflowStep],
        parallel: bool = False
    ) -> WorkflowResult:
        """
        Execute a multi-step workflow.
        
        Args:
            steps: List of workflow steps
            parallel: If True, run independent steps in parallel
        
        Returns:
            WorkflowResult with execution status and results
        """
        results = {}
        errors = {}
        status = WorkflowStatus.RUNNING
        
        # Build dependency graph
        step_map = {step.name: step for step in steps}
        executed = set()
        
        async def execute_step(step: WorkflowStep):
            """Execute a single step."""
            nonlocal status
            if step.name in executed:
                return results.get(step.name)
            
            # Check dependencies
            for dep in step.depends_on:
                if dep not in executed:
                    if dep in step_map:
                        await execute_step(step_map[dep])
                    else:
                        errors[step.name] = f"Missing dependency: {dep}"
                        return None
            
            # Execute step
            try:
                if step.server == "anima":
                    result = await self._execute_anima_step(step)
                elif step.server == "unitares":
                    result = await self._execute_unitares_step(step)
                else:
                    raise ValueError(f"Unknown server: {step.server}")
                
                results[step.name] = result
                executed.add(step.name)
                return result
                
            except Exception as e:
                errors[step.name] = str(e)
                status = WorkflowStatus.PARTIAL
                return None
        
        # Execute steps
        if parallel:
            # Run independent steps in parallel
            tasks = []
            for step in steps:
                # Check if dependencies are satisfied
                deps_satisfied = all(dep in executed for dep in step.depends_on)
                if deps_satisfied:
                    tasks.append(execute_step(step))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Sequential execution
            for step in steps:
                await execute_step(step)
        
        # Determine final status
        if len(errors) == len(steps):
            status = WorkflowStatus.FAILED
        elif len(errors) > 0:
            status = WorkflowStatus.PARTIAL
        else:
            status = WorkflowStatus.SUCCESS
        
        # Generate summary
        summary = f"Workflow {status.value}: {len(results)}/{len(steps)} steps succeeded"
        if errors:
            summary += f", {len(errors)} errors"
        
        return WorkflowResult(
            status=status,
            steps=results,
            errors=errors,
            summary=summary
        )
    
    async def _execute_anima_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a step on anima-mcp server."""
        if step.tool == "get_state":
            return await self.get_lumen_state()
        elif step.tool == "read_sensors":
            # Try shared memory first, then fallback to sensors
            readings, _ = self._get_readings_and_anima()
            if readings:
                return {"sensors": readings.to_dict()}
            return {"error": "Sensors not available"}
        elif step.tool == "get_identity":
            if self._anima_store:
                identity = self._anima_store.get_identity()
                return {
                    "name": identity.name,
                    "id": identity.creature_id,
                    "awakenings": identity.total_awakenings,
                    "alive_seconds": identity.total_alive_seconds,
                }
            return {"error": "Store not available"}
        elif step.tool == "get_calibration":
            # Get current calibration
            from .config import get_calibration
            calibration = get_calibration()
            return {
                "calibration": calibration.to_dict(),
            }
        else:
            raise ValueError(f"Unknown anima tool: {step.tool}")
    
    async def _execute_unitares_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a step on unitares-governance server."""
        if step.tool == "check_governance":
            anima = step.arguments.get("anima")
            readings = step.arguments.get("readings")
            eisv = step.arguments.get("eisv")
            return await self.check_governance(anima, readings, eisv)
        elif step.tool == "process_agent_update":
            # Forward to UNITARES bridge
            if self._bridge:
                return await self._bridge.process_agent_update(**step.arguments)
            return {"error": "Bridge not available"}
        else:
            raise ValueError(f"Unknown unitares tool: {step.tool}")
    
    async def workflow_check_state_and_governance(self) -> Dict[str, Any]:
        """
        Common workflow: Check Lumen's state, then check governance.

        Returns:
            Combined result with state and governance decision
        """
        state = await self.get_lumen_state()

        if "error" in state:
            return state

        # Call check_governance without pre-computed values
        # It will read fresh sensors and compute anima/eisv internally
        governance = await self.check_governance()

        return {
            "state": state,
            "governance": governance,
            "timestamp": state.get("timestamp"),
        }
    
    async def workflow_monitor_and_govern(self, interval: float = 60.0) -> Dict[str, Any]:
        """
        Continuous workflow: Monitor Lumen and check governance periodically.
        
        Args:
            interval: Seconds between checks
        
        Returns:
            Latest monitoring result
        """
        result = await self.workflow_check_state_and_governance()
        
        # Could extend this to run continuously and yield results
        return result


# Global orchestrator instance
_orchestrator: Optional[UnifiedWorkflowOrchestrator] = None


def get_orchestrator(
    unitares_url: Optional[str] = None,
    anima_store=None,
    anima_sensors=None
) -> UnifiedWorkflowOrchestrator:
    """Get global workflow orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = UnifiedWorkflowOrchestrator(
            unitares_url=unitares_url,
            anima_store=anima_store,
            anima_sensors=anima_sensors
        )
    return _orchestrator
