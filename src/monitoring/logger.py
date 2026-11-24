import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_LOG_LOCK = threading.Lock()


def get_log_file_path() -> str:
    log_dir = os.getenv("INFERENCE_LOG_DIR", "logs")
    return os.path.join(log_dir, "inference_requests.jsonl")


def log_inference_batch(
    instances: List[Dict[str, Any]],
    predictions: List[str],
    model_uri: Optional[str] = None,
    latency_ms: Optional[float] = None,
) -> None:
    """
    Registra in formato JSON Lines le feature di ingresso, i metadati di sottogruppo
    (compagnia e aeroporto) e le predizioni prodotte dall'endpoint /classify.
    """
    if os.getenv("DISABLE_INFERENCE_LOGGING", "false").lower() == "true":
        return

    log_file = get_log_file_path()
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    batch_id = str(uuid.uuid4())
    resolved_model_uri = model_uri or os.getenv("MODEL_URI", "unknown")

    lines = []
    for idx, (instance, pred) in enumerate(zip(instances, predictions)):
        entry = {
            "timestamp": timestamp,
            "batch_id": batch_id,
            "record_index": idx,
            "model_uri": resolved_model_uri,
            "latency_ms": latency_ms,
            # Campi chiave promossi in primo livello per query/aggregazioni rapide
            "carrier": instance.get("OP_UNIQUE_CARRIER"),
            "origin": instance.get("ORIGIN"),
            "dest": instance.get("DEST"),
            "fl_date": instance.get("FL_DATE"),
            "predicted_severity": str(pred),
            # Payload grezzo completo per analisi Data Drift e join con delayed labels
            "features": instance,
        }
        lines.append(json.dumps(entry, default=str))

    payload = "\n".join(lines) + "\n"
    with _LOG_LOCK:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(payload)