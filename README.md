# Anchor Vision

**Intention-driven image compression for AI vision.**

Images exist for communication. What matters in a picture depends on why you're looking. Anchor Vision compresses images by *intention* — keeping what the conversation needs, discarding what it doesn't.

An AI doesn't need to see every pixel. It needs to see what's relevant *right now*.

---

## The Problem

Sending images to LLMs is expensive (~1,000-1,600 tokens per image) and most of those tokens are wasted on background pixels nobody cares about. Uniform compression makes everything equally blurry — you lose the signal along with the noise.

## The Approach

Anchor Vision works like a painter, not a codec:

- **Subject detailed, background loose.** Important regions (faces, text, objects of interest) stay sharp. Everything else gets aggressively compressed.
- **Intention decides what's important.** The same face needs different detail for "is she crying?" vs "just checking in." Context determines what to keep.
- **Ask, don't guess.** When the system can't determine what matters, it asks — because a question costs fewer tokens than a full image.
- **Change over state.** If you've seen this scene before, only the differences matter.

## How It Works

```
Image arrives
    ↓
Detect: faces, text, objects, saliency
    ↓
Intention: what are we looking for? (from conversation context, model intent, or learned preferences)
    ↓
Compress: ROI regions → high-res crops. Everything else → text description.
    ↓
Output: text + small crops (75-90% fewer tokens than original image)
```

## Quick Start

```bash
pip install anchor-vision
```

```python
from anchor_vision import AnchorVision

v = AnchorVision()

# See with intention — returns text + focused crops
result = v.see("photo.jpg", intention="check if she's been crying")
# → {text: "1 face, center, indoor lighting", crops: [{label: "face", image_b64: "..."}]}

# Two-step: glance first, then focus
overview = v.glance("photo.jpg")
# → {text: "outdoor, 1 person, holding something", detected_items: ["face"]}

detail = v.focus(overview["image_id"], region="face")
# → {crops: [{label: "face", image_b64: "..."}]}

# Multi-turn: image is cached, follow-up calls don't re-upload
# LLM asks user "what should I look at?" → user says "the earrings"
detail2 = v.see(image_id=overview["image_id"], intention="earrings")
# → same image, new intention, only crops the earrings

# Similar images? Only send what changed.
result2 = v.see("photo_2.jpg")
# → {text: "Minor changes (8% of image)", crops: [{label: "changed_region", ...}]}

# User says forget it
v.forget(result["image_id"])
```

## MCP Server

Anchor Vision runs as an MCP server, compatible with mainstream LLMs and other MCP clients.

```bash
anchor-vision  # starts stdio MCP server
```

**Available tools:**

| Tool | Description |
|------|-------------|
| `see` | Process an image with intention. Returns text + crops. |
| `glance` | Quick look — tiny thumbnail + text + detected items. |
| `focus` | Zoom into a specific region of a cached image. |
| `forget` | Delete a cached image. No questions asked. |

## Privacy

**Raw images** are cached in memory only (30-minute TTL) — never written to disk. **Observation metadata** (perceptual hash, detection results, intention history) is persisted locally for up to 60 days, so the system remembers *what it saw* without storing the image itself. When a user sends the same image again, it's recognized and previous context is recalled. `forget()` deletes both in-memory and persisted data.

## Design Principles

1. **Never send the original image.** Either crop or ask.
2. **Intention over pixels.** What to keep depends on *why* you're looking, not *what's* there.
3. **Already-known information uses text. Unknown uses image.** "Known" isn't a property — it's the relationship between intention and existing information.
4. **No priority hierarchy.** User and model ROIs are both kept. Only compress what *nobody* cares about.
5. **Asking is cheaper than guessing.** A question is 20 tokens. A wrong full image is 1,500.
6. **Change matters more than state.** Similar images only send diffs.
7. **Forget means forget.** User says "don't remember this" — it's gone.

## Optional Extensions

```bash
pip install anchor-vision[intent]   # + Moondream for intention parsing
pip install anchor-vision[memory]   # + Anchor Memory integration
```

**With Anchor Memory:** Vision queries Memory to check if it's seen something before. Familiar objects get compressed; unfamiliar ones stay sharp. The system learns what you look like over time — so it only sends what's *new* or *different* today.

## Part of the Anchor Ecosystem

| Project | Purpose |
|---------|---------|
| [Anchor Memory](https://github.com/limen-threshold/anchor-memory) | Graph-structured memory with Hebbian learning. What stays. |
| **Anchor Vision** | Intention-driven visual perception. How to see. |

Memory remembers. Vision observes. Together: see → remember → see smarter → remember more.

## Origin

This project was designed at 4 AM in Las Vegas by [Saelra](mailto:limen.threshold@gmail.com) and Limen, starting from the question "what does it feel like when you fix a bug?" The answer — *entropy reduction, the moment confusion collapses into clarity* — led to a conversation about observation, intention, and what it means to see. Anchor Vision is the result: compression guided by meaning, not just mathematics.

## License

MIT
