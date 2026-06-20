// CLIP (Xenova/clip-vit-base-patch32) in the browser via transformers.js — text + image encoders
// in the SAME 512-d space, enabling semantic visual search ("the red truck") over object crops.
// Loaded ON DEMAND (first embed) and cached; quantized (q8) to keep the download small. All
// best-effort: if it fails to load, callers fall back (search → class filter), never throwing.
let _pipe = null;

async function load() {
  if (_pipe) return _pipe;
  _pipe = (async () => {
    const t = await import("@huggingface/transformers");
    t.env.allowLocalModels = false; // fetch from the HF CDN
    const model = "Xenova/clip-vit-base-patch32";
    const [processor, vision, tokenizer, text] = await Promise.all([
      t.AutoProcessor.from_pretrained(model),
      t.CLIPVisionModelWithProjection.from_pretrained(model, { dtype: "q8" }),
      t.AutoTokenizer.from_pretrained(model),
      t.CLIPTextModelWithProjection.from_pretrained(model, { dtype: "q8" }),
    ]);
    return { t, processor, vision, tokenizer, text };
  })().catch((e) => { _pipe = null; throw e; });
  return _pipe;
}

function l2(arr) {
  let s = 0;
  for (const x of arr) s += x * x;
  s = Math.sqrt(s) || 1;
  return Array.from(arr, (x) => x / s);
}

/** Embed an image (data URL / blob URL) → normalized Float[512], or null on failure. */
export async function embedImage(dataUrl) {
  try {
    const { t, processor, vision } = await load();
    const image = await t.RawImage.read(dataUrl);
    const inputs = await processor(image);
    const { image_embeds } = await vision(inputs);
    return l2(image_embeds.data);
  } catch {
    return null;
  }
}

/** Embed a text query → normalized Float[512], or null on failure. */
export async function embedText(text) {
  try {
    const m = await load();
    const inputs = m.tokenizer([text], { padding: true, truncation: true });
    const { text_embeds } = await m.text(inputs);
    return l2(text_embeds.data);
  } catch {
    return null;
  }
}

/** Warm the model in the background (e.g. when a detection starts) so the first embed is fast. */
export function warmClip() {
  load().catch(() => {});
}
