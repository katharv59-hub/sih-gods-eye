"""FastAPI Application — Phase 10.4 (SIH 26187).

Thin read-only REST API layer. Endpoints delegate to existing
ReasoningEngine/ToolDispatcher internally — no bypass.

Endpoints:
  GET /cameras, /events, /alerts, /alerts/{id},
  /subjects/{id}, /vehicles/{id}, /plates/{plate},
  /timeline/{subject}, /evidence/{id}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from gods_eye.observability.logger import get_logger

_log = get_logger("api.app")


def create_app(
    event_store: Optional[Any] = None,
    graph_store: Optional[Any] = None,
    alert_store: Optional[Any] = None,
    evidence_manager: Optional[Any] = None,
    reasoning_engine: Optional[Any] = None,
    tool_dispatcher: Optional[Any] = None,
) -> Any:
    """Create and configure the FastAPI application.

    Args:
        event_store: SQLiteEventStore instance.
        graph_store: SQLiteGraphStore instance.
        alert_store: SQLiteAlertStore instance.
        evidence_manager: EvidenceManager instance.
        reasoning_engine: ReasoningEngine instance.
        tool_dispatcher: ToolDispatcher instance.

    Returns:
        FastAPI application instance.
    """
    try:
        from fastapi import FastAPI, HTTPException, Header, Query  # type: ignore[import-untyped]
        from fastapi.staticfiles import StaticFiles  # type: ignore[import-untyped]
        from fastapi.responses import HTMLResponse, JSONResponse  # type: ignore[import-untyped]
    except ImportError:
        _log.error("fastapi_not_installed", msg="Install fastapi: pip install fastapi uvicorn")
        raise ImportError("FastAPI is required for the API module. Install with: pip install fastapi uvicorn")

    from gods_eye.api.auth import verify_token

    app = FastAPI(
        title="God's Eye Command-and-Control API",
        description="Read-only situational intelligence API — SIH 26187",
        version="1.0.0",
    )

    # Mount static files for dashboard
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def _auth(authorization: Optional[str] = None) -> None:
        """Verify API token from Authorization header."""
        token = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]
        if not verify_token(token):
            raise HTTPException(status_code=401, detail="Invalid API token")

    @app.get("/")
    async def root() -> dict[str, str]:
        return {
            "service": "God's Eye API",
            "version": "1.0.0",
            "status": "operational",
        }

    @app.get("/dashboard")
    async def dashboard() -> HTMLResponse:
        dashboard_path = static_dir / "dashboard.html"
        if dashboard_path.exists():
            return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>Dashboard not available</h1>", status_code=404)

    @app.get("/cameras")
    async def get_cameras(authorization: Optional[str] = Header(None)) -> JSONResponse:
        _auth(authorization)
        # Return configured cameras from graph store
        cameras: list[dict[str, Any]] = []
        if graph_store is not None:
            try:
                with graph_store._lock:
                    rows = graph_store._conn.execute(
                        "SELECT camera_id, status, metadata FROM node_cameras"
                    ).fetchall()
                for row in rows:
                    cameras.append({
                        "camera_id": row[0],
                        "status": row[1],
                    })
            except Exception:
                pass
        return JSONResponse(content={"cameras": cameras})

    @app.get("/events")
    async def get_events(
        authorization: Optional[str] = Header(None),
        limit: int = Query(50, ge=1, le=500),
        event_type: Optional[str] = Query(None),
    ) -> JSONResponse:
        _auth(authorization)
        events: list[dict[str, Any]] = []
        if event_store is not None:
            try:
                if event_type:
                    from gods_eye.schemas.event import EventType
                    et = EventType(event_type)
                    raw = event_store.query_events(event_type=et, limit=limit)
                else:
                    raw = event_store.query_events(limit=limit)
                events = [
                    {
                        "event_id": e.event_id,
                        "event_type": e.event_type.value,
                        "camera_id": e.camera_id,
                        "timestamp_ns": e.timestamp_ns,
                        "confidence": e.confidence,
                        "explanation": e.explanation,
                    }
                    for e in raw
                ]
            except Exception:
                pass
        return JSONResponse(content={"events": events, "count": len(events)})

    @app.get("/alerts")
    async def get_alerts(
        authorization: Optional[str] = Header(None),
        severity: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=500),
    ) -> JSONResponse:
        _auth(authorization)
        alerts: list[dict[str, Any]] = []
        if alert_store is not None:
            try:
                raw = alert_store.get_alerts_by_severity(severity=severity, limit=limit)
                alerts = [a.to_dict() for a in raw]
            except Exception:
                pass
        return JSONResponse(content={"alerts": alerts, "count": len(alerts)})

    @app.get("/alerts/{alert_id}")
    async def get_alert_by_id(
        alert_id: str,
        authorization: Optional[str] = Header(None),
    ) -> JSONResponse:
        _auth(authorization)
        if alert_store is None:
            raise HTTPException(status_code=503, detail="Alert store not available")
        alert = alert_store.get_alert(alert_id)
        if alert is None:
            raise HTTPException(status_code=404, detail="Alert not found")
        return JSONResponse(content=alert.to_dict())

    @app.get("/subjects/{subject_id}")
    async def get_subject(
        subject_id: str,
        authorization: Optional[str] = Header(None),
    ) -> JSONResponse:
        _auth(authorization)
        result: dict[str, Any] = {"subject_id": subject_id}
        if graph_store is not None:
            try:
                with graph_store._lock:
                    row = graph_store._conn.execute(
                        "SELECT * FROM node_identities WHERE global_id = ?",
                        (subject_id,),
                    ).fetchone()
                if row:
                    result["found"] = True
                    result["first_seen_ns"] = row[1] if len(row) > 1 else None
                    result["last_seen_ns"] = row[2] if len(row) > 2 else None
                else:
                    result["found"] = False
            except Exception:
                result["found"] = False
        return JSONResponse(content=result)

    @app.get("/vehicles/{vehicle_id}")
    async def get_vehicle(
        vehicle_id: str,
        authorization: Optional[str] = Header(None),
    ) -> JSONResponse:
        _auth(authorization)
        # Query events for vehicle detections
        result: dict[str, Any] = {"vehicle_id": vehicle_id, "events": []}
        if event_store is not None:
            try:
                from gods_eye.schemas.event import EventType
                raw = event_store.query_events(event_type=EventType.VEHICLE_DETECTED, limit=100)
                result["events"] = [
                    {"event_id": e.event_id, "timestamp_ns": e.timestamp_ns}
                    for e in raw
                    if getattr(e, "metadata", {}).get("vehicle_track_id") == vehicle_id
                ]
            except Exception:
                pass
        return JSONResponse(content=result)

    @app.get("/plates/{plate}")
    async def get_plate_history(
        plate: str,
        authorization: Optional[str] = Header(None),
    ) -> JSONResponse:
        _auth(authorization)
        # Use Tool 13 through dispatcher if available
        result: dict[str, Any] = {"plate": plate, "readings": []}
        if tool_dispatcher is not None:
            try:
                from gods_eye.schemas.reasoning import ToolCall
                import uuid
                tc = ToolCall(
                    call_id=str(uuid.uuid4()),
                    tool_name="get_plate_history",
                    arguments={"plate_text": plate},
                )
                tr = tool_dispatcher.dispatch(tc)
                if tr.success:
                    result["readings"] = tr.data if tr.data else []
            except Exception:
                pass
        return JSONResponse(content=result)

    @app.get("/timeline/{subject}")
    async def get_timeline(
        subject: str,
        authorization: Optional[str] = Header(None),
    ) -> JSONResponse:
        _auth(authorization)
        # Internally call ReasoningEngine/ToolDispatcher
        result: dict[str, Any] = {"subject": subject, "timeline": []}
        if tool_dispatcher is not None:
            try:
                from gods_eye.schemas.reasoning import ToolCall
                import uuid
                tc = ToolCall(
                    call_id=str(uuid.uuid4()),
                    tool_name="get_identity_timeline",
                    arguments={"global_id": subject},
                )
                tr = tool_dispatcher.dispatch(tc)
                if tr.success:
                    result["timeline"] = tr.data if tr.data else []
            except Exception:
                pass
        return JSONResponse(content=result)

    @app.get("/evidence/{alert_id}")
    async def get_evidence(
        alert_id: str,
        authorization: Optional[str] = Header(None),
    ) -> JSONResponse:
        _auth(authorization)
        evidence: list[dict[str, Any]] = []
        if evidence_manager is not None:
            evidence = evidence_manager.get_evidence(alert_id)
        return JSONResponse(content={"alert_id": alert_id, "evidence": evidence})

    return app
