"""FastAPI Application — Phase 10.4 (SIH 26187).

Thin read-only REST API layer. Endpoints delegate to existing
ReasoningEngine/ToolDispatcher internally — no bypass.

Endpoints:
  GET /cameras, /events, /alerts, /alerts/{id},
  /subjects/{id}, /vehicles/{id}, /plates/{plate},
  /timeline/{subject}, /evidence/{id}
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Optional

from gods_eye.observability.logger import get_logger

try:
    from fastapi import UploadFile
except ImportError:
    UploadFile = Any  # type: ignore[misc,assignment]

_log = get_logger("api.app")


def create_app(
    event_store: Optional[Any] = None,
    graph_store: Optional[Any] = None,
    alert_store: Optional[Any] = None,
    evidence_manager: Optional[Any] = None,
    reasoning_engine: Optional[Any] = None,
    tool_dispatcher: Optional[Any] = None,
    face_recognizer: Optional[Any] = None,
) -> Any:
    """Create and configure the FastAPI application.

    Args:
        event_store: SQLiteEventStore instance.
        graph_store: SQLiteGraphStore instance.
        alert_store: AlertStore instance.
        evidence_manager: EvidenceManager instance.
        reasoning_engine: ReasoningEngine instance.
        tool_dispatcher: ToolDispatcher instance.

    Returns:
        FastAPI application instance.
    """
    try:
        from fastapi import FastAPI, File, Form, HTTPException, Header, Query, UploadFile  # type: ignore[import-untyped]
        from fastapi.staticfiles import StaticFiles  # type: ignore[import-untyped]
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse  # type: ignore[import-untyped]
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

    def _auth(authorization: Optional[str] = None, token_param: Optional[str] = None) -> None:
        """Verify API token from Authorization header or query parameter."""
        token = token_param
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

    @app.get("/feeds/cam/{cam_id}")
    async def get_camera_feed(cam_id: str) -> StreamingResponse:
        """Stream real-time surveillance video footage (MJPEG) for camera viewports."""
        import cv2

        cam_key = cam_id.upper().replace("-", "").strip()
        repo_root = Path(__file__).resolve().parent.parent.parent

        video_map = {
            "CAM01": repo_root / "tests" / "data" / "demo_outputs" / "mot17_04_medium_density_annotated.mp4",
            "CAM02": repo_root / "tests" / "data" / "demo_outputs" / "mot17_05_high_density_annotated.mp4",
            "CAM03": repo_root / "tests" / "data" / "demo_outputs" / "mot17_09_low_density_annotated.mp4",
            "CAM04": repo_root / "tests" / "data" / "demo_videos" / "mot17_04_medium_density.mp4",
        }

        video_path = video_map.get(cam_key, video_map["CAM01"])
        if not video_path.exists():
            candidates = list((repo_root / "tests" / "data" / "demo_outputs").glob("*.mp4"))
            if candidates:
                video_path = candidates[0]
            else:
                raise HTTPException(status_code=404, detail="Camera video feed not found")

        is_night_cam = (cam_key == "CAM04")

        async def mjpeg_frame_generator():
            cap = cv2.VideoCapture(str(video_path))
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            break

                    h, w = frame.shape[:2]
                    # Resize to 640px width for fast encode & smooth browser FPS
                    if w > 640:
                        target_h = int(h * (640 / w))
                        frame = cv2.resize(frame, (640, target_h), interpolation=cv2.INTER_LINEAR)

                    if is_night_cam:
                        # Apply green phosphor night-IR surveillance grade palette
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        colored = cv2.applyColorMap(gray, cv2.COLORMAP_SUMMER)
                        frame = cv2.addWeighted(colored, 0.85, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.15, 0)

                    success, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if not success:
                        await asyncio.sleep(0.04)
                        continue

                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                    )
                    await asyncio.sleep(0.033)  # ~30 FPS
            finally:
                cap.release()

        return StreamingResponse(
            mjpeg_frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

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
            except Exception as exc:
                _log.error("get_cameras_failed", error=str(exc))
                raise HTTPException(status_code=500, detail="Internal server error")
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
            et = None
            if event_type:
                from gods_eye.schemas.event import EventType
                try:
                    et = EventType(event_type)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid event_type: {event_type}")
            try:
                if et is not None:
                    raw = event_store.query_events(event_type=et, limit=limit)
                else:
                    raw = event_store.query_events(limit=limit)
                events = [
                    {
                        "event_id": e.event_id,
                        "event_type": e.event_type.value,
                        "global_id": e.global_id,
                        "camera_id": e.camera_id,
                        "timestamp_ns": e.timestamp_ns,
                        "confidence": e.confidence,
                        "explanation": e.explanation,
                        "metadata": e.metadata,
                    }
                    for e in raw
                ]
            except HTTPException:
                raise
            except Exception as exc:
                _log.error("get_events_failed", error=str(exc))
                raise HTTPException(status_code=500, detail="Internal server error")
        return JSONResponse(content={"events": events, "count": len(events)})

    @app.delete("/events")
    async def delete_events_endpoint(
        event_type: Optional[str] = Query(None),
        camera_id: Optional[str] = Query(None),
        authorization: Optional[str] = Header(None),
    ) -> JSONResponse:
        """Clear/delete events matching filter criteria."""
        _auth(authorization)
        from gods_eye.schemas.event import EventType
        target_ev_type: Optional[EventType] = None
        if event_type:
            try:
                target_ev_type = EventType(event_type)
            except ValueError:
                try:
                    target_ev_type = EventType[event_type.upper()]
                except KeyError:
                    raise HTTPException(status_code=400, detail=f"Invalid event_type: {event_type}")

        deleted_count = 0
        if event_store is not None:
            try:
                if hasattr(event_store, "delete_events"):
                    deleted_count = event_store.delete_events(event_type=target_ev_type, camera_id=camera_id)
            except Exception as exc:
                _log.error("delete_events_failed", error=str(exc))
                raise HTTPException(status_code=500, detail="Internal server error")

        return JSONResponse(
            content={
                "status": "cleared",
                "deleted_count": deleted_count,
                "event_type": event_type,
            }
        )

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
            except Exception as exc:
                _log.error("get_alerts_failed", error=str(exc))
                raise HTTPException(status_code=500, detail="Internal server error")
        return JSONResponse(content={"alerts": alerts, "count": len(alerts)})

    @app.post("/alerts/simulate", status_code=201)
    async def simulate_alert(
        alert_type: str = Query("intrusion"),
        camera_id: str = Query("CAM-02"),
        authorization: Optional[str] = Header(None),
    ) -> JSONResponse:
        """Inject an operational threat alert into AlertStore and EventStore for live demonstration."""
        _auth(authorization)
        import time, uuid
        from gods_eye.schemas.alert import Alert, AlertStatus
        from gods_eye.schemas.event import Event, EventType

        ts = time.time_ns()
        alert_id = f"alt_sim_{uuid.uuid4().hex[:8]}"

        type_map = {
            "intrusion": ("CRITICAL", "VAULT_GEOFENCE", "CAM-02", "SUBJ_9941", "Vault Perimeter: Point-in-polygon intrusion in restricted zone with loitering > 5.2σ"),
            "plate": ("HIGH", "NORTH_GATE", "CAM-01", "VEH_8812", "Plate Sighting: Registered watchlist vehicle DL01AB1234 detected at North Gate"),
            "dwell": ("MEDIUM", "ATM_ZONE", "CAM-03", "PERSON_10", "Loitering Alert: Dwell threshold exceeded in ATM Corridor (142s > 60s baseline)"),
            "night": ("HIGH", "DOCK_ZONE", "CAM-04", "SUBJ_0411", "Night Movement: Off-hours transition detected in Loading Dock during NIGHT_DARK"),
        }
        severity, zone, default_cam, subj, desc = type_map.get(alert_type.lower(), type_map["intrusion"])
        cam = camera_id or default_cam

        if alert_store is not None:
            alert = Alert(
                alert_id=alert_id,
                severity=severity,
                status=AlertStatus.ACTIVE,
                subject_ref=subj,
                zone_id=zone,
                camera_id=cam,
                timestamp_ns=ts,
                risk_signal_id=f"sig_{uuid.uuid4().hex[:6]}",
                explanation=desc,
                metadata={"simulated": True},
            )
            alert_store.append_alert(alert)
        elif event_store is not None:
            ev = Event(
                event_id=alert_id,
                event_type=EventType.ALERT_CREATED,
                camera_id=cam,
                zone_id=zone,
                timestamp_ns=ts,
                confidence=0.98,
                explanation=desc,
                metadata={"simulated": True, "alert_severity": severity, "alert_status": "active"},
            )
            event_store.append(ev)

        return JSONResponse(
            status_code=201,
            content={
                "status": "created",
                "alert_id": alert_id,
                "severity": severity,
                "description": desc,
                "camera_id": cam,
                "timestamp_ns": ts,
            }
        )

    @app.get("/alerts/{alert_id}")
    async def get_alert_by_id(
        alert_id: str,
        authorization: Optional[str] = Header(None),
    ) -> JSONResponse:
        _auth(authorization)
        if alert_store is None:
            raise HTTPException(status_code=503, detail="Alert store not available")
        try:
            alert = alert_store.get_alert(alert_id)
        except Exception as exc:
            _log.error("get_alert_by_id_failed", alert_id=alert_id, error=str(exc))
            raise HTTPException(status_code=500, detail="Internal server error")
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
                    result["primary_camera_id"] = row[3] if len(row) > 3 else None
                    result["state"] = row[4] if len(row) > 4 else None
                    if len(row) > 5 and row[5]:
                        try:
                            import json
                            result["metadata"] = json.loads(row[5]) if isinstance(row[5], str) else row[5]
                        except Exception:
                            result["metadata"] = row[5]
                else:
                    result["found"] = False
            except Exception as exc:
                _log.error("get_subject_failed", subject_id=subject_id, error=str(exc))
                raise HTTPException(status_code=500, detail="Internal server error")
        else:
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
            except Exception as exc:
                _log.error("get_vehicle_failed", vehicle_id=vehicle_id, error=str(exc))
                raise HTTPException(status_code=500, detail="Internal server error")
        return JSONResponse(content=result)

    @app.get("/plates/{plate}")
    async def get_plate_history(
        plate: str,
        authorization: Optional[str] = Header(None),
    ) -> JSONResponse:
        _auth(authorization)
        clean_plate = plate.strip()
        if not clean_plate:
            raise HTTPException(status_code=400, detail="Plate text must not be empty")

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
                    result["readings"] = tr.data if tr.data is not None else []
                else:
                    _log.error("get_plate_history_tool_failed", plate=plate, error=tr.error_message)
                    raise HTTPException(status_code=500, detail="Internal server error")
            except HTTPException:
                raise
            except Exception as exc:
                _log.error("get_plate_history_failed", plate=plate, error=str(exc))
                raise HTTPException(status_code=500, detail="Internal server error")
        return JSONResponse(content=result)

    @app.get("/timeline/{subject}")
    async def get_timeline(
        subject: str,
        authorization: Optional[str] = Header(None),
    ) -> JSONResponse:
        _auth(authorization)
        clean_subject = subject.strip()
        if not clean_subject:
            raise HTTPException(status_code=400, detail="Subject ID must not be empty")

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
                    result["timeline"] = tr.data if tr.data is not None else []
                else:
                    _log.error("get_timeline_tool_failed", subject=subject, error=tr.error_message)
                    raise HTTPException(status_code=500, detail="Internal server error")
            except HTTPException:
                raise
            except Exception as exc:
                _log.error("get_timeline_failed", subject=subject, error=str(exc))
                raise HTTPException(status_code=500, detail="Internal server error")
        return JSONResponse(content=result)

    @app.get("/evidence/{alert_id}")
    async def get_evidence(
        alert_id: str,
        authorization: Optional[str] = Header(None),
    ) -> JSONResponse:
        _auth(authorization)
        evidence: list[dict[str, Any]] = []
        if evidence_manager is not None:
            try:
                evidence = evidence_manager.get_evidence(alert_id)
                for item in evidence:
                    ev_id = item.get("evidence_id")
                    if ev_id:
                        if item.get("has_snapshot") or "snapshot_path" in item:
                            item["snapshot_url"] = f"/evidence/file/{alert_id}/{ev_id}/snapshot.jpg"
                        if item.get("has_face_crop") or "face_crop_path" in item:
                            item["face_crop_url"] = f"/evidence/file/{alert_id}/{ev_id}/face_crop.jpg"
            except Exception as exc:
                _log.error("get_evidence_failed", alert_id=alert_id, error=str(exc))
                raise HTTPException(status_code=500, detail="Internal server error")
        return JSONResponse(content={"alert_id": alert_id, "evidence": evidence})

    @app.get("/evidence/file/{alert_id}/{evidence_id}/{filename}")
    async def get_evidence_file(
        alert_id: str,
        evidence_id: str,
        filename: str,
        authorization: Optional[str] = Header(None),
        token: Optional[str] = Query(None),
    ) -> FileResponse:
        """Safely serve captured evidence images (snapshot.jpg, face_crop.jpg)."""
        _auth(authorization, token_param=token)
        allowed_filenames = {"snapshot.jpg", "face_crop.jpg"}
        if filename not in allowed_filenames:
            raise HTTPException(
                status_code=400,
                detail=f"Filename not allowed. Must be one of: {', '.join(sorted(allowed_filenames))}",
            )

        for param, name in [(alert_id, "alert_id"), (evidence_id, "evidence_id"), (filename, "filename")]:
            if ".." in param or "/" in param or "\\" in param or "%" in param:
                raise HTTPException(status_code=400, detail=f"Invalid characters in {name}")

        base_path_val = (
            getattr(evidence_manager, "base_path", getattr(evidence_manager, "_base_path", "data/evidence"))
            if evidence_manager is not None
            else "data/evidence"
        )
        evidence_root = Path(base_path_val).resolve()
        target_file = (evidence_root / alert_id / evidence_id / filename).resolve()

        # Fallback: if alert_id is identical to evidence_id or folder nesting differs, search within alert_id dir
        if not target_file.is_file() and (evidence_root / alert_id).is_dir():
            for candidate_dir in (evidence_root / alert_id).iterdir():
                if candidate_dir.is_dir():
                    candidate_file = candidate_dir / filename
                    if candidate_file.is_file():
                        target_file = candidate_file.resolve()
                        break

        try:
            target_file.relative_to(evidence_root)
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied: Path traversal detected")

        if not target_file.is_file():
            raise HTTPException(status_code=404, detail="Evidence file not found")

        return FileResponse(str(target_file), media_type="image/jpeg")

    @app.post("/faces/enroll", status_code=201)
    async def enroll_face(
        name: str = Form(...),
        file: UploadFile = File(...),
        authorization: Optional[str] = Header(None),
    ) -> JSONResponse:
        _auth(authorization)
        clean_name = name.strip()
        if not clean_name:
            raise HTTPException(status_code=400, detail="Name must not be empty")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        import cv2
        import numpy as np
        import uuid
        import time

        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file or unreadable format")

        recognizer = face_recognizer
        if recognizer is None:
            try:
                from gods_eye.face.recognizer import FaceRecognizer
                recognizer = FaceRecognizer()
            except Exception as exc:
                _log.error("face_recognizer_unavailable", error=str(exc))
                raise HTTPException(status_code=503, detail="Face recognition model unavailable")

        faces = recognizer.detect_faces(frame)
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected in uploaded image")
        if len(faces) > 1:
            raise HTTPException(status_code=400, detail="Multiple faces detected; exactly one face required")

        bbox, conf, face_data = faces[0]
        embedding = recognizer.extract_embedding(frame, face_data)

        person_id = f"person_{uuid.uuid4().hex[:12]}"

        faces_dir = Path("data/faces/enrolled")
        faces_dir.mkdir(parents=True, exist_ok=True)
        photo_path = faces_dir / f"{person_id}.jpg"
        cv2.imwrite(str(photo_path), frame)

        target_store = graph_store
        if target_store is None:
            from gods_eye.memory.graph_store import SQLiteGraphStore
            target_store = SQLiteGraphStore("data/graph.db")

        ts = time.time_ns()
        target_store.enroll_face(
            person_id=person_id,
            name=clean_name,
            embedding=embedding,
            photo_path=str(photo_path),
            timestamp_ns=ts,
        )

        return JSONResponse(
            status_code=201,
            content={
                "person_id": person_id,
                "name": clean_name,
                "created_ns": ts,
                "dimension": int(embedding.shape[0]),
                "detection_confidence": conf,
            },
        )

    @app.get("/faces/enrolled")
    async def get_enrolled_faces(
        authorization: Optional[str] = Header(None),
    ) -> JSONResponse:
        _auth(authorization)
        target_store = graph_store
        if target_store is None:
            from gods_eye.memory.graph_store import SQLiteGraphStore
            target_store = SQLiteGraphStore("data/graph.db")

        records = target_store.get_enrolled_faces()
        enrolled_list = [
            {
                "person_id": r.person_id,
                "name": r.name,
                "created_ns": r.created_ns,
                "dimension": r.dimension,
                "photo_path": r.photo_path,
                "photo_url": f"/faces/photo/{r.person_id}",
            }
            for r in records
        ]
        return JSONResponse(content={"enrolled": enrolled_list, "count": len(enrolled_list)})

    @app.delete("/faces/enrolled/{person_id}")
    async def delete_enrolled_face(
        person_id: str,
        authorization: Optional[str] = Header(None),
    ) -> JSONResponse:
        """Delete an enrolled person record and their reference photo."""
        _auth(authorization)
        if ".." in person_id or "/" in person_id or "\\" in person_id or "%" in person_id:
            raise HTTPException(status_code=400, detail="Invalid characters in person_id")

        target_store = graph_store
        if target_store is None:
            from gods_eye.memory.graph_store import SQLiteGraphStore
            target_store = SQLiteGraphStore("data/graph.db")

        deleted = target_store.delete_enrolled_face(person_id)

        # Remove stored photo file if present
        enrolled_root = Path("data/faces/enrolled").resolve()
        clean_pid = person_id if person_id.endswith(".jpg") else f"{person_id}.jpg"
        photo_file = (enrolled_root / clean_pid).resolve()
        try:
            if photo_file.relative_to(enrolled_root) and photo_file.is_file():
                photo_file.unlink()
        except Exception:
            pass

        if not deleted:
            raise HTTPException(status_code=404, detail=f"Enrolled person {person_id} not found")

        return JSONResponse(
            content={
                "status": "deleted",
                "person_id": person_id,
                "message": f"Successfully deleted enrolled person {person_id}",
            }
        )

    @app.get("/faces/photo/{person_id}")
    async def get_enrolled_face_photo(
        person_id: str,
        authorization: Optional[str] = Header(None),
        token: Optional[str] = Query(None),
    ) -> FileResponse:
        """Safely serve enrolled person reference photograph."""
        _auth(authorization, token_param=token)
        if ".." in person_id or "/" in person_id or "\\" in person_id or "%" in person_id:
            raise HTTPException(status_code=400, detail="Invalid characters in person_id")

        enrolled_root = Path("data/faces/enrolled").resolve()
        clean_pid = person_id if person_id.endswith(".jpg") else f"{person_id}.jpg"
        photo_file = (enrolled_root / clean_pid).resolve()

        try:
            photo_file.relative_to(enrolled_root)
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied: Path traversal detected")

        if not photo_file.is_file():
            raise HTTPException(status_code=404, detail="Enrolled face photo not found")

        return FileResponse(str(photo_file), media_type="image/jpeg")

    @app.post("/faces/recognize")
    async def recognize_face_endpoint(
        file: UploadFile = File(...),
        authorization: Optional[str] = Header(None),
    ) -> JSONResponse:
        _auth(authorization)
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        import cv2
        import numpy as np

        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file or unreadable format")

        recognizer = face_recognizer
        if recognizer is None:
            try:
                from gods_eye.face.recognizer import FaceRecognizer
                recognizer = FaceRecognizer()
            except Exception as exc:
                _log.error("face_recognizer_unavailable", error=str(exc))
                raise HTTPException(status_code=503, detail="Face recognition model unavailable")

        target_store = graph_store
        if target_store is None:
            from gods_eye.memory.graph_store import SQLiteGraphStore
            target_store = SQLiteGraphStore("data/graph.db")

        gallery = target_store.get_enrolled_faces()
        try:
            res = recognizer.recognize(frame, gallery)
        except ValueError as val_err:
            raise HTTPException(status_code=400, detail=str(val_err))

        return JSONResponse(
            content={
                "status": res.status,
                "person_id": res.person_id,
                "name": res.name,
                "similarity": res.similarity,
                "detection_confidence": res.detection_confidence,
                "bbox": [res.bbox.x1, res.bbox.y1, res.bbox.x2, res.bbox.y2],
            }
        )

    # Live Engine Cache for real-time webcam frame streams
    _live_engine_cache: dict[str, Any] = {}

    def _get_live_engine() -> Optional[Any]:
        if "engine" in _live_engine_cache:
            return _live_engine_cache["engine"]
        if event_store is not None and evidence_manager is not None:
            try:
                from gods_eye.face.live_engine import LiveFaceEngine
                from gods_eye.face.recognizer import FaceRecognizer
                from gods_eye.memory.graph_store import SQLiteGraphStore

                r = face_recognizer or FaceRecognizer()
                gs = graph_store or SQLiteGraphStore("data/graph.db")
                engine = LiveFaceEngine(
                    recognizer=r,
                    graph_store=gs,
                    event_store=event_store,
                    evidence_manager=evidence_manager,
                    throttle_window_s=3.0,
                )
                _live_engine_cache["engine"] = engine
                return engine
            except Exception as e:
                _log.warning("live_engine_init_failed", error=str(e))
        return None

    @app.post("/faces/recognize_frame")
    async def recognize_frame_endpoint(
        file: UploadFile = File(...),
        authorization: Optional[str] = Header(None),
        persist: bool = Query(True),
    ) -> JSONResponse:
        """Process a live webcam frame: detect all faces, identify enrolled persons, and return recognition overlay data."""
        _auth(authorization)
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded frame is empty")

        import cv2
        import numpy as np

        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file or unreadable format")

        rec = face_recognizer
        if rec is None:
            try:
                from gods_eye.face.recognizer import FaceRecognizer
                rec = FaceRecognizer()
            except Exception as exc:
                _log.error("face_recognizer_unavailable", error=str(exc))
                raise HTTPException(status_code=503, detail="Face recognition model unavailable")

        # Check if live engine can process and emit canonical events
        live_eng = _get_live_engine() if persist else None
        if live_eng is not None:
            try:
                results = live_eng.process_frame(frame, camera_id="WEBCAM")
                faces_data = [
                    {
                        "status": r.status,
                        "person_id": r.person_id,
                        "name": r.name,
                        "similarity": float(r.similarity),
                        "detection_confidence": float(r.detection_confidence),
                        "bbox": [float(r.bbox.x1), float(r.bbox.y1), float(r.bbox.x2), float(r.bbox.y2)],
                    }
                    for r in results
                ]
                return JSONResponse(
                    content={
                        "faces": faces_data,
                        "count": len(faces_data),
                        "frame_width": int(frame.shape[1]),
                        "frame_height": int(frame.shape[0]),
                    }
                )
            except Exception as exc:
                _log.warning("live_engine_process_fallback", error=str(exc))

        # Direct recognition without EventStore
        target_store = graph_store
        if target_store is None:
            from gods_eye.memory.graph_store import SQLiteGraphStore
            target_store = SQLiteGraphStore("data/graph.db")

        gallery = target_store.get_enrolled_faces()
        detected = rec.detect_faces(frame)
        faces_data = []

        for bbox, conf, face_data in detected:
            try:
                query_emb = rec.extract_embedding(frame, face_data)
            except Exception:
                continue

            best_sim = -1.0
            best_rec = None
            for record in gallery:
                ref_emb = record.embedding if hasattr(record, "embedding") else record["embedding"]
                if isinstance(ref_emb, (bytes, bytearray)):
                    ref_emb = np.frombuffer(ref_emb, dtype=np.float32)
                elif isinstance(ref_emb, list):
                    ref_emb = np.array(ref_emb, dtype=np.float32)
                sim = rec.match(ref_emb, query_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_rec = record

            best_sim_bounded = max(0.0, best_sim)
            if best_rec is not None and best_sim >= rec.recognition_threshold:
                st = "KNOWN"
                pid = str(best_rec.person_id if hasattr(best_rec, "person_id") else best_rec["person_id"])
                pname = str(best_rec.name if hasattr(best_rec, "name") else best_rec["name"])
            else:
                st = "UNKNOWN"
                pid = None
                pname = None

            faces_data.append({
                "status": st,
                "person_id": pid,
                "name": pname,
                "similarity": float(best_sim_bounded),
                "detection_confidence": float(conf),
                "bbox": [float(bbox.x1), float(bbox.y1), float(bbox.x2), float(bbox.y2)],
            })

        return JSONResponse(
            content={
                "faces": faces_data,
                "count": len(faces_data),
                "frame_width": int(frame.shape[1]),
                "frame_height": int(frame.shape[0]),
            }
        )

    return app


# Default application instance for ASGI servers (e.g. uvicorn gods_eye.api.app:app)
try:
    from gods_eye.config.settings import Settings
    from gods_eye.events.event_store import SQLiteEventStore
    from gods_eye.memory.graph_store import SQLiteGraphStore
    from gods_eye.alerts.alert_store import AlertStore
    from gods_eye.evidence.evidence_manager import EvidenceManager
    from gods_eye.reasoning.tools import ToolDispatcher
    from gods_eye.face.recognizer import FaceRecognizer

    _settings = Settings.from_env()
    _default_event_store = SQLiteEventStore(db_path=_settings.event_store_path)
    _default_graph_store = SQLiteGraphStore(db_path=_settings.graph_store_path)
    _default_alert_store = AlertStore(event_store=_default_event_store)
    _default_evidence_mgr = EvidenceManager(base_path=_settings.evidence_base_path)
    _default_dispatcher = ToolDispatcher(event_store=_default_event_store)
    try:
        _default_face_rec = FaceRecognizer()
    except Exception:
        _default_face_rec = None

    app = create_app(
        event_store=_default_event_store,
        graph_store=_default_graph_store,
        alert_store=_default_alert_store,
        evidence_manager=_default_evidence_mgr,
        tool_dispatcher=_default_dispatcher,
        face_recognizer=_default_face_rec,
    )
except Exception as _init_exc:
    _log.warning("api_default_instance_fallback", error=str(_init_exc))
    app = create_app()


