"""Minimal dependency-free MCP stdio transport for the Codex HA-VLN adapter.

MCP uses one JSON-RPC message per line over stdio. Keep stdout exclusively
for protocol messages: Habitat libraries may print during tool execution.
"""
from contextlib import redirect_stdout
import json
import sys


TOOLS = [
    {
        "name": "observe",
        "description": (
            "Return the current RGB and short observation metadata; add a Depth image "
            "only for a near obstacle or motion stall. Repeated turns may add a compact view comparison. No hidden map."
        ),
        "inputSchema": {"type": "object", "properties": {},
                        "additionalProperties": False},
    },
    {
        "name": "step",
        "description": (
            "Execute an ordered sequence of HA-VLN actions in the live Habitat "
            "episode. Legal actions: STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, "
            "LOOK_UP, LOOK_DOWN. STOP must be last. All actions consume the "
            "shared episode budget. Verified stalled or ambiguous forward movement can stop "
            "the remaining batch; inspect the feedback before trying again."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array", "minItems": 1,
                    "items": {"type": "string", "enum": [
                        "STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT",
                        "LOOK_UP", "LOOK_DOWN",
                    ]},
                },
            },
            "required": ["actions"],
            "additionalProperties": False,
        },
    },
]


class MCPServer:
    def __init__(self, episode, reader=None, writer=None):
        self.episode = episode
        self.reader = reader if reader is not None else sys.stdin
        self.writer = writer if writer is not None else sys.stdout

    def _reply(self, payload):
        self.writer.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        self.writer.flush()

    def _dispatch(self, method, params):
        if method == "initialize":
            # Compatible tool wire shapes are shared across these versions.
            protocol = params.get("protocolVersion", "2025-06-18")
            return {
                "protocolVersion": protocol,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "havln-codex", "version": "0.1.0"},
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": TOOLS}
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be an object")
            if name == "observe":
                if arguments:
                    raise ValueError("observe takes no arguments")
                with redirect_stdout(sys.stderr):
                    info, rgb_data, depth_data = self.episode.observe()
                content = [
                    {"type": "text", "text": json.dumps(info, ensure_ascii=False)},
                    {"type": "image", "data": rgb_data, "mimeType": "image/png"},
                ]
                if depth_data is not None:
                    content.append({"type": "image", "data": depth_data,
                                    "mimeType": "image/png"})
                comparison = getattr(self.episode, "scan_montage", None)
                if comparison is not None:
                    content.append({"type": "image", "data": comparison,
                                    "mimeType": "image/png"})
                return {"content": content, "isError": False}
            if name == "step":
                with redirect_stdout(sys.stderr):
                    result = self.episode.step(arguments.get("actions"))
                return {
                    "content": [{"type": "text", "text":
                                 json.dumps(result, ensure_ascii=False)}],
                    "isError": result["status"] == "TOOL_ERROR",
                }
            raise ValueError("Unknown navigation tool: " + str(name))
        raise LookupError("Unsupported MCP method: " + str(method))

    def serve(self):
        for line in self.reader:
            if not line.strip():
                continue
            request_id = None
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("Expected a JSON-RPC object")
                request_id = request.get("id")
                method = request.get("method")
                if not isinstance(method, str):
                    raise ValueError("Missing JSON-RPC method")
                if method.startswith("notifications/"):
                    continue
                # A notification has no id and must not receive a reply.
                if "id" not in request:
                    continue
                result = self._dispatch(method, request.get("params") or {})
                self._reply({"jsonrpc": "2.0", "id": request_id, "result": result})
            except Exception as error:
                if request_id is not None:
                    self._reply({
                        "jsonrpc": "2.0", "id": request_id,
                        "error": {"code": -32603, "message": str(error)},
                    })
                else:
                    sys.stderr.write("MCP input error: " + str(error) + "\n")
