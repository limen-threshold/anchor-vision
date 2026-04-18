"""Anchor Vision MCP Server — expose vision tools via MCP protocol."""
import json
import sys
import os
from typing import Any

from .vision import AnchorVision

# MCP protocol implementation (stdio-based)
# Compatible with Claude Code, claude.ai, and other MCP clients


class MCPServer:
    def __init__(self):
        self.vision = AnchorVision()
        self.tools = {
            "see": {
                "description": "Process an image with intention-driven compression. Returns text description + focused crops instead of the full image. First call: provide image_path. Follow-up calls on same image: provide image_id.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to image file (first call)",
                        },
                        "image_id": {
                            "type": "string",
                            "description": "Cached image ID (follow-up calls)",
                        },
                        "intention": {
                            "type": "string",
                            "description": "Why are you looking? e.g. 'check if she is crying', 'read the text on the sign', 'see what changed'",
                        },
                    },
                },
            },
            "glance": {
                "description": "Quick look at an image — returns tiny thumbnail + text description + list of detected items. Use this first, then focus() on specific areas.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to image file",
                        },
                    },
                    "required": ["image_path"],
                },
            },
            "focus": {
                "description": "Focus on a specific region of a previously seen image. Returns high-res crop of that region.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "image_id": {
                            "type": "string",
                            "description": "Image ID from a previous see() or glance() call",
                        },
                        "region": {
                            "type": "string",
                            "description": "What to focus on: 'the face', 'top left', 'the text', etc.",
                        },
                    },
                    "required": ["image_id"],
                },
            },
            "forget": {
                "description": "Forget a cached image. User says 'don't remember this' — it's gone.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "image_id": {
                            "type": "string",
                            "description": "Image ID to forget",
                        },
                    },
                    "required": ["image_id"],
                },
            },
        }

    def handle_request(self, request: dict) -> dict:
        method = request.get("method", "")
        req_id = request.get("id")

        if method == "initialize":
            return self._response(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "anchor-vision", "version": "0.1.0"},
            })

        elif method == "tools/list":
            tools_list = []
            for name, spec in self.tools.items():
                tools_list.append({
                    "name": name,
                    "description": spec["description"],
                    "inputSchema": spec["inputSchema"],
                })
            return self._response(req_id, {"tools": tools_list})

        elif method == "tools/call":
            tool_name = request["params"]["name"]
            args = request["params"].get("arguments", {})
            return self._call_tool(req_id, tool_name, args)

        elif method == "notifications/initialized":
            return None  # No response needed

        return self._error(req_id, -32601, f"Unknown method: {method}")

    def _call_tool(self, req_id: Any, name: str, args: dict) -> dict:
        try:
            if name == "see":
                result = self.vision.see(
                    image_path=args.get("image_path"),
                    image_id=args.get("image_id"),
                    intention=args.get("intention"),
                )
            elif name == "glance":
                result = self.vision.glance(args["image_path"])
            elif name == "focus":
                result = self.vision.focus(
                    args["image_id"],
                    region=args.get("region"),
                )
            elif name == "forget":
                result = self.vision.forget(args["image_id"])
            else:
                return self._error(req_id, -32602, f"Unknown tool: {name}")

            # Format crops for display
            content = self._format_result(result)
            return self._response(req_id, {"content": content})

        except Exception as e:
            return self._response(req_id, {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "isError": True,
            })

    def _format_result(self, result: dict) -> list:
        """Format result as MCP content blocks."""
        content = []

        # Text description
        text_parts = []
        if result.get("text"):
            text_parts.append(result["text"])
        if result.get("suggestion"):
            text_parts.append(f"Suggestion: {result['suggestion']}")
        if result.get("image_id"):
            text_parts.append(f"[image_id: {result['image_id']}]")
        if result.get("detected_items"):
            text_parts.append(f"Detected: {', '.join(result['detected_items'])}")
        if result.get("diff_from"):
            text_parts.append(f"(compared to previous: {result['diff_from']})")

        if text_parts:
            content.append({"type": "text", "text": "\n".join(text_parts)})

        # Thumbnail
        if result.get("thumbnail_b64"):
            content.append({
                "type": "image",
                "data": result["thumbnail_b64"],
                "mimeType": "image/jpeg",
            })

        # Crops
        for crop in result.get("crops", []):
            content.append({
                "type": "text",
                "text": f"[{crop['label']}] (from {crop.get('source', 'auto')})",
            })
            content.append({
                "type": "image",
                "data": crop["image_b64"],
                "mimeType": "image/jpeg",
            })

        # Uncertain regions
        for unc in result.get("uncertain", []):
            content.append({
                "type": "text",
                "text": f"[uncertain: {unc['label']} at {unc['bbox']}] — use focus() to see this area",
            })

        return content

    def _response(self, req_id: Any, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _error(self, req_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    def run(self):
        """Run the MCP server on stdio."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                if response is not None:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
            except json.JSONDecodeError:
                err = self._error(None, -32700, "Parse error")
                sys.stdout.write(json.dumps(err) + "\n")
                sys.stdout.flush()


def main():
    server = MCPServer()
    server.run()


if __name__ == "__main__":
    main()
